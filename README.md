# `bevy_compute_readback`

[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/Katsutoshii/bevy_compute_readback#license)
[![Crates.io](https://img.shields.io/crates/v/bevy_compute_readback.svg)](https://crates.io/crates/bevy_compute_readback)
[![Docs](https://docs.rs/bevy_compute_readback/badge.svg)](https://docs.rs/bevy_compute_readback/latest/bevy_compute_readback/)

Crate to abstract away the boilerplate of creating compute shaders with readback in the Bevy game engine.

This based on the GPU readback example ([gpu_readback.rs](https://github.com/bevyengine/bevy/blob/main/examples/shader/gpu_readback.rs)).

## Usage

```rs
use bevy_compute_readback::{
    ComputeShader, ComputeShaderPlugin, ComputeShaderReadback, ReadbackLimit
};

// Create a resource to store your shader inputs and derive AsBindGroup.
#[derive(AsBindGroup, Resource, Clone, Debug, ExtractResource)]
pub struct CustomComputeShader {
    #[storage_texture(0, image_format=Rgba8Uint, access=WriteOnly)]
    texture: Handle<Image>,
}

// Implement ComputeShader for your custom shader
impl ComputeShader for CustomComputeShader {
    fn compute_shader() -> ShaderRef {
        "shaders/some_custom_shader.wgsl".into()
    }
    fn workgroup_size() -> UVec3 {
        UVec3::new(64, 64, 1)
    }
    fn readbacks(&self) -> impl Bundle {
        // Define which inputs should be read back to CPU.
        Readback::texture(self.texture.clone())
    }
    fn on_readback(trigger: Trigger<ReadbackComplete>, mut world: DeferredWorld) {
        // Do something with the readback data.
    }
}

/// Create the readback entity to receive updates from CustomComputeShader.
fn setup(mut commands: Commands) {
    commands.spawn(ComputeShaderReadback::<CustomComputeShader> {
        limit: ReadbackLimit::Finite(1),
        ..default()
    });
}

/// In your app, add ComputeShaderPlugin.
app.add_plugins((
    ComputeShaderPlugin::<CustomComputeShader>::default(),
));
```

See `examples` for a working demo.

## Bevy support table

| bevy | bevy_compute_readback |
| ---- | --------------------- |
| 0.16 | 0.1.0                 |
