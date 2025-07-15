# `bevy_compute_readback`

[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/Katsutoshii/bevy_compute_readback#license)
[![Crates.io](https://img.shields.io/crates/v/bevy_compute_readback.svg)](https://crates.io/crates/bevy_compute_readback)
[![Docs](https://docs.rs/bevy_compute_readback/badge.svg)](https://docs.rs/bevy_compute_readback/latest/bevy_compute_readback/)

Crate to abstract away the boilerplate of creating compute shaders with readback in the Bevy game engine.

This based on the GPU readback example ([gpu_readback.rs](https://github.com/bevyengine/bevy/blob/main/examples/shader/gpu_readback.rs)).

## Usage

```rs
use bevy_compute_readback::{
    ComputeShader, ComputeShaderPlugin, ReadbackLimit
};

/// Custom compute shader input.
#[derive(AsBindGroup, Resource, Clone, Debug, ExtractResource)]
pub struct CustomComputeShader {
    // Texture for the GPU to write to.
    #[storage_texture(0, image_format=Rgba32Float, access=WriteOnly)]
    texture: Handle<Image>,
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
    fn on_readback(trigger: Trigger<ReadbackComplete>, mut world: DeferredWorld) {
        // ...
    }
}

fn main() {
    App::new()
        .add_plugins((
            ComputeShaderPlugin::<CustomComputeShader> {
                limit: ReadbackLimit::Finite(1),
                remove_on_complete: false,
                ..default()
            },
        ))
        .run();
}
```

See `examples` for a working demo.

## Bevy support table

| bevy | bevy_compute_readback |
| ---- | --------------------- |
| 0.16 | 0.1.1                 |
