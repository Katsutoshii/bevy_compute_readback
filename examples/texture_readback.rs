//! Example to demonstrate reading texture data back to CPU from a compute shader.
//! Press Space to cycle through different inputs to the shader to demonstrate reactivity.
use bevy::{
    asset::RenderAssetUsages,
    ecs::world::DeferredWorld,
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{AsBindGroup, Extent3d, TextureDimension, TextureFormat, TextureUsages},
    },
    shader::ShaderRef,
};
use bevy_compute_readback::{ComputeShader, ComputeShaderPlugin, ReadbackLimit};
use image::DynamicImage;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            // Initialize compute shader pipeline.
            ComputeShaderPlugin::<CustomComputeShader> {
                limit: ReadbackLimit::Finite(1),
                remove_on_complete: false,
                ..default()
            },
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .add_systems(Startup, setup)
        .add_systems(Update, reload_on_space)
        .run();
}

/// Visualize the compute shader output as a sprite.
fn setup(mut commands: Commands, shader: Res<CustomComputeShader>) {
    commands.spawn(Camera2d);
    commands.spawn((
        Sprite::from_image(shader.readback_texture.clone()),
        Transform {
            scale: Vec3::splat(5.0),
            ..default()
        },
    ));
}

/// Update the inputs to the shader on Space.
fn reload_on_space(
    inputs: Res<ButtonInput<KeyCode>>,
    mut shader: ResMut<CustomComputeShader>,
    mut count: Local<usize>,
) {
    if inputs.just_pressed(KeyCode::Space) {
        info!("space pressed");
        *count = (*count + 1) % 4;
        shader.color = match *count {
            0 => LinearRgba::new(1.0, 1.0, 1.0, 1.0),
            1 => LinearRgba::new(0.2, 1.0, 1.0, 1.0),
            2 => LinearRgba::new(1.0, 0.2, 1.0, 1.0),
            3 => LinearRgba::new(1.0, 1.0, 0.2, 1.0),
            _ => unreachable!(),
        };
    }
}

// Custom compute shader input.
#[derive(AsBindGroup, Resource, Clone, Debug, ExtractResource)]
pub struct CustomComputeShader {
    // Texture for the GPU to write to.
    #[storage_texture(0, image_format=Rgba32Float, access=WriteOnly)]
    texture: Handle<Image>,

    // Input color
    #[uniform(1)]
    pub color: LinearRgba,

    // Texture where data will be read back to from GPU.
    // We only need this because we want to render the read back texture.
    readback_texture: Handle<Image>,
}
impl ComputeShader for CustomComputeShader {
    /// Path to your compute shader WGSL file.
    fn compute_shader() -> ShaderRef {
        "shaders/texture_readback.wgsl".into()
    }
    /// Workgroup size for the compute shader.
    fn workgroup_size() -> UVec3 {
        UVec3::new(64, 64, 1)
    }
    /// Indicate which buffer/texture should be read back to CPU.
    fn readback(&self) -> Option<Readback> {
        Some(Readback::texture(self.texture.clone()))
    }
    /// Handle readback events.
    fn on_readback(trigger: On<ReadbackComplete>, mut world: DeferredWorld) {
        // Copy readback buffer to the render texture so we can see it.
        // Then save it as a PNG.
        let image_handle = world.resource::<Self>().readback_texture.clone();
        if let Some(image) = world.resource_mut::<Assets<Image>>().get_mut(&image_handle) {
            image.data = Some(trigger.event().data.clone());
            info!("Readback");
            if let Ok(DynamicImage::ImageRgba32F(rgba)) = image.clone().try_into_dynamic() {
                let _ = rgba.save("target/readback_output.png");
            }
        } else {
            warn!("Handle not ready: {:?}", image_handle);
        }
    }
}
impl FromWorld for CustomComputeShader {
    /// Initialize the shader with empty textures.
    fn from_world(world: &mut World) -> Self {
        let workgroup_size = Self::workgroup_size();
        let size = Extent3d {
            width: workgroup_size.x,
            height: workgroup_size.y,
            depth_or_array_layers: workgroup_size.z,
        };
        let pixel = 0f32.to_le_bytes().repeat(4);
        let mut image = Image::new_fill(
            size,
            TextureDimension::D2,
            &pixel,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING;
        Self {
            texture: world.add_asset(image.clone()),
            readback_texture: world.add_asset(image),
            color: LinearRgba::new(1.0, 1.0, 1.0, 1.0),
        }
    }
}
