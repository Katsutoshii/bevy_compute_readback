//! Library to simplify compute shader readbacks.
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use bevy_app::{App, Plugin, Startup};
use bevy_asset::DirectAssetAccessExt;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    observer::Trigger,
    query::With,
    resource::Resource,
    schedule::{
        Condition, IntoScheduleConfigs,
        common_conditions::{not, resource_changed, resource_exists, resource_exists_and_changed},
    },
    system::{Commands, Query, Res, ResMut, StaticSystemParam},
    world::{DeferredWorld, FromWorld, World},
};
use bevy_math::UVec3;
use bevy_render::{
    ExtractSchedule, MainWorld, Render, RenderApp, RenderSet,
    extract_resource::{ExtractResource, ExtractResourcePlugin, extract_resource},
    gpu_readback::{Readback, ReadbackComplete},
    render_graph::{self, RenderGraph, RenderLabel},
    render_resource::{
        AsBindGroup, BindGroup, BindGroupLayout, CachedComputePipelineId, CachedPipelineState,
        ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderRef,
    },
    renderer::{RenderContext, RenderDevice},
};
use bevy_state::{
    app::AppExtStates,
    state::{NextState, OnEnter, States},
};

/// Plugin to create all the required systems for using a custom compute shader.
pub struct ComputeShaderPlugin<S: ComputeShader> {
    pub limit: ReadbackLimit,
    pub remove_on_complete: bool,
    pub _marker: PhantomData<S>,
}
impl<S: ComputeShader> Default for ComputeShaderPlugin<S> {
    fn default() -> Self {
        Self {
            limit: ReadbackLimit::default(),
            remove_on_complete: false,
            _marker: PhantomData,
        }
    }
}
impl<S: ComputeShader> Plugin for ComputeShaderPlugin<S> {
    fn build(&self, app: &mut App) {
        app.init_resource::<S>()
            .add_plugins(ExtractResourcePlugin::<S>::default())
            .init_state::<ComputeNodeState<S>>()
            .add_systems(
                OnEnter(ComputeNodeState::<S>::from(ComputeNodeStatus::Ready)),
                ComputeShaderReadback::<S>::on_shader_ready,
            )
            .add_systems(
                OnEnter(ComputeNodeState::<S>::from(ComputeNodeStatus::Completed)),
                ComputeShaderReadback::<S>::on_shader_complete,
            )
            .add_systems(Startup, ComputeShaderReadback::<S>::spawn);
    }

    fn finish(&self, app: &mut App) {
        // Add the compute shader resources and systems to the render app.
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<ComputePipeline<S>>()
            .init_resource::<ComputeNodeState<S>>()
            .add_systems(
                ExtractSchedule,
                ComputeNode::<S>::reset_on_change
                    .run_if(resource_exists_and_changed::<S>)
                    .after(extract_resource::<S>),
            )
            .add_systems(
                ExtractSchedule,
                ComputeNodeState::<S>::extract_to_main
                    .run_if(resource_changed::<ComputeNodeState<S>>),
            )
            .add_systems(
                Render,
                (S::prepare_bind_group)
                    .chain()
                    .in_set(RenderSet::PrepareBindGroups)
                    .run_if(
                        not(resource_exists::<ComputeShaderBindGroup<S>>).or(resource_changed::<S>),
                    ),
            );

        // Add the compute node as a top level node to the render graph
        // This means it will only execute once per frame
        render_app
            .world_mut()
            .resource_mut::<RenderGraph>()
            .add_node(
                ComputeNodeLabel::<S>::default(),
                ComputeNode::<S> {
                    limit: self.limit,
                    ..Default::default()
                },
            );

        // If the compute node should be removed on completion, schedule the removal systems.
        if self.remove_on_complete {
            render_app.add_systems(
                ExtractSchedule,
                ComputeNodeLabel::<S>::remove_on_complete
                    .run_if(resource_changed::<ComputeNodeState<S>>),
            );
        }
    }
}

/// How many readbacks should be sent per initialization of the shader.
#[derive(Default, Debug, Copy, Clone)]
pub enum ReadbackLimit {
    /// No limit, readback will continue indefinitely.
    #[default]
    Infinite,
    /// Finite readback limit, measured in number of frames.
    Finite(usize),
}

/// Component that receives readback events from the compute shader.
#[derive(Component)]
pub struct ComputeShaderReadback<S: ComputeShader> {
    pub _marker: PhantomData<S>,
}
impl<S: ComputeShader> Default for ComputeShaderReadback<S> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<S: ComputeShader> ComputeShaderReadback<S> {
    /// Spawn the readback observer on startup.
    fn spawn(mut commands: Commands) {
        commands.spawn(Self::default()).observe(S::on_readback);
    }
    /// Insert GPU readback component only when the shader is ready.
    fn on_shader_ready(
        mut commands: Commands,
        compute_shader: Res<S>,
        mut compute_shader_readbacks: Query<Entity, With<Self>>,
    ) {
        for entity in compute_shader_readbacks.iter_mut() {
            if let Some(readback) = compute_shader.readback() {
                commands.entity(entity).insert(readback);
            }
        }
    }
    /// Disable the shader when it's done.
    fn on_shader_complete(
        mut commands: Commands,
        mut compute_shader_readbacks: Query<Entity, With<Self>>,
    ) {
        for entity in compute_shader_readbacks.iter_mut() {
            commands.entity(entity).remove::<Readback>();
        }
    }
}

/// Trait to implement for a custom compute shader.
pub trait ComputeShader: AsBindGroup + Clone + Debug + FromWorld + ExtractResource {
    /// Asset path or handle to the shader.
    fn compute_shader() -> ShaderRef;
    /// Workgroup size.
    fn workgroup_size() -> UVec3;
    /// Optional bind group preparation.
    fn prepare_bind_group(
        mut commands: Commands,
        pipeline: Res<ComputePipeline<Self>>,
        render_device: Res<RenderDevice>,
        input: Res<Self>,
        param: StaticSystemParam<<Self as AsBindGroup>::Param>,
    ) {
        let bind_group = input
            .as_bind_group(&pipeline.layout, &render_device, &mut param.into_inner())
            .unwrap();
        commands.insert_resource(ComputeShaderBindGroup::<Self> {
            bind_group: bind_group.bind_group,
            _marker: PhantomData,
        });
    }
    /// Optional readbacks.
    fn readback(&self) -> Option<Readback> {
        None
    }
    /// Optional processing on readback. Could write back to the CPU buffer, etc.
    fn on_readback(_trigger: Trigger<ReadbackComplete>, mut _world: DeferredWorld) {}
}

/// Stores prepared bind group data for the compute shader.
#[derive(Resource)]
pub struct ComputeShaderBindGroup<S: ComputeShader> {
    pub bind_group: BindGroup,
    pub _marker: PhantomData<S>,
}

/// Enum representing possible compute node states.
#[derive(Default, Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ComputeNodeStatus {
    #[default]
    Loading,
    Init,
    Ready,
    Completed,
    Error,
}
/// Tracks compute node state.
/// In render world, this is stored as a resource which is later extracted to main.
/// In main world, this is a state so systems can react to state entry.
#[derive(States, Resource, Clone, Copy, Debug)]
pub struct ComputeNodeState<S: ComputeShader> {
    status: ComputeNodeStatus,
    _marker: PhantomData<S>,
}
impl<S: ComputeShader> Hash for ComputeNodeState<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.status.hash(state);
    }
}
impl<S: ComputeShader> PartialEq for ComputeNodeState<S> {
    fn eq(&self, other: &Self) -> bool {
        self.status == other.status
    }
}
impl<S: ComputeShader> Eq for ComputeNodeState<S> {}
impl<S: ComputeShader> From<ComputeNodeStatus> for ComputeNodeState<S> {
    fn from(value: ComputeNodeStatus) -> Self {
        Self {
            status: value,
            _marker: PhantomData,
        }
    }
}
impl<S: ComputeShader> Default for ComputeNodeState<S> {
    fn default() -> Self {
        Self {
            status: ComputeNodeStatus::default(),
            _marker: PhantomData,
        }
    }
}
impl<S: ComputeShader> ComputeNodeState<S> {
    /// Extracts compute node state resource into a state
    /// that systems can react to in the main world.
    fn extract_to_main(compute_state: Res<ComputeNodeState<S>>, mut world: ResMut<MainWorld>) {
        world
            .resource_mut::<NextState<ComputeNodeState<S>>>()
            .set(compute_state.clone());
    }
}

/// Defines the pipeline for the compute shader.
#[derive(Resource)]
pub struct ComputePipeline<S: ComputeShader> {
    pub layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
    _marker: PhantomData<S>,
}
impl<S: ComputeShader> FromWorld for ComputePipeline<S> {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = S::bind_group_layout(render_device);
        let shader = match S::compute_shader() {
            ShaderRef::Default => panic!("Must define compute_shader."),
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => world.load_asset(path),
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("GPU readback compute shader".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: Vec::new(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });
        Self {
            layout,
            pipeline,
            _marker: PhantomData,
        }
    }
}

/// Label to identify the node in the render graph.
#[derive(Debug, Clone, RenderLabel)]
struct ComputeNodeLabel<S: ComputeShader> {
    _marker: PhantomData<S>,
}
impl<S: ComputeShader> Default for ComputeNodeLabel<S> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<S: ComputeShader> PartialEq for ComputeNodeLabel<S> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<S: ComputeShader> Eq for ComputeNodeLabel<S> {}
impl<S: ComputeShader> Hash for ComputeNodeLabel<S> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}
impl<S: ComputeShader> ComputeNodeLabel<S> {
    fn remove_on_complete(mut render_graph: ResMut<RenderGraph>, state: Res<ComputeNodeState<S>>) {
        if state.status == ComputeNodeStatus::Completed {
            let _ = render_graph.remove_node(Self::default());
        }
    }
}

/// The node that will execute the compute shader.
/// Updates `ComputeNodeState<S>` in the `RenderWorld`.
struct ComputeNode<S: ComputeShader> {
    status: ComputeNodeStatus,
    limit: ReadbackLimit,
    count: usize,
    _marker: PhantomData<S>,
}
impl<S: ComputeShader> Default for ComputeNode<S> {
    fn default() -> Self {
        Self {
            status: ComputeNodeStatus::default(),
            limit: ReadbackLimit::Infinite,
            count: 0,
            _marker: PhantomData,
        }
    }
}
impl<S: ComputeShader> ComputeNode<S> {
    /// When the input shader is changed, reset.
    fn reset_on_change(
        mut render_graph: ResMut<RenderGraph>,
        mut state: ResMut<ComputeNodeState<S>>,
    ) {
        let Ok(node) = render_graph.get_node_mut::<Self>(ComputeNodeLabel::<S>::default()) else {
            return;
        };
        node.count = 0;
        node.status = ComputeNodeStatus::Loading;
        *state = ComputeNodeState {
            status: ComputeNodeStatus::Loading,
            ..Default::default()
        };
    }
}
impl<S: ComputeShader> render_graph::Node for ComputeNode<S> {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<ComputePipeline<S>>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let next_status = match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline) {
            CachedPipelineState::Ok(_) => match (self.status, self.limit) {
                (ComputeNodeStatus::Completed, _) => ComputeNodeStatus::Completed,
                (_, ReadbackLimit::Finite(limit)) => {
                    if self.count < limit {
                        self.count += 1;
                        ComputeNodeStatus::Ready
                    } else {
                        self.count = 0;
                        ComputeNodeStatus::Completed
                    }
                }
                _ => ComputeNodeStatus::Ready,
            },
            CachedPipelineState::Creating(_) => ComputeNodeStatus::Loading,
            CachedPipelineState::Queued => ComputeNodeStatus::Loading,
            CachedPipelineState::Err(_) => ComputeNodeStatus::Error,
        };

        if self.status != next_status {
            self.status = next_status;
            world.resource_mut::<ComputeNodeState<S>>().status = next_status;
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ComputePipeline<S>>();
        let bind_group = &world.resource::<ComputeShaderBindGroup<S>>().bind_group;
        if self.status == ComputeNodeStatus::Ready {
            if let Some(init_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
                let workgroup_size = S::workgroup_size();
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("GPU readback compute pass"),
                            ..Default::default()
                        });
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(workgroup_size.x, workgroup_size.y, workgroup_size.z);
            }
        }
        Ok(())
    }
}
