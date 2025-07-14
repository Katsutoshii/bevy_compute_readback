use bevy::{
    asset::RenderAssetUsages,
    ecs::world::DeferredWorld,
    prelude::*,
    render::gpu_readback::ReadbackComplete,
    render::{
        extract_resource::ExtractResource,
        gpu_readback::Readback,
        render_resource::{
            AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat, TextureUsages,
        },
    },
};
use bevy_compute_readback::{
    ComputeShader, ComputeShaderPlugin, ComputeShaderReadback, ReadbackLimit,
};
use image::{Rgba, RgbaImage};

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins,
        // Initialize compute shader pipeline.
        ComputeShaderPlugin::<CustomComputeShader> {
            limit: ReadbackLimit::Finite(1),
            ..default()
        },
    ))
    .insert_resource(ClearColor(Color::BLACK))
    .add_systems(Startup, setup)
    .run();
}

/// Create the readback entity to receive updates from CustomComputeShader.
fn setup(mut commands: Commands) {
    commands.spawn(ComputeShaderReadback::<CustomComputeShader>::default());
}

/// Converts an image to RGBA so it can be saved as a PNG.
fn image_to_rgba(image: &Image) -> RgbaImage {
    let mut rgba = RgbaImage::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let srgba: Srgba = image.get_color_at(x, y).unwrap().to_srgba();
            *rgba.get_pixel_mut(x, y) = Rgba(srgba.to_u8_array());
        }
    }
    rgba
}

// Custom compute shader input.
#[derive(AsBindGroup, Resource, Clone, Debug, ExtractResource)]
pub struct CustomComputeShader {
    #[storage_texture(0, image_format=Rgba8Uint, access=WriteOnly)]
    texture: Handle<Image>,

    // Where data will be read back to.
    readback_texture: Handle<Image>,
}
impl ComputeShader for CustomComputeShader {
    fn compute_shader() -> ShaderRef {
        "shaders/texture_readback.wgsl".into()
    }
    fn workgroup_size() -> UVec3 {
        UVec3::new(64, 64, 1)
    }
    fn readback(&self) -> Option<Readback> {
        Some(Readback::texture(self.texture.clone()))
    }
    fn on_readback(trigger: Trigger<ReadbackComplete>, mut world: DeferredWorld) {
        let data: Vec<u8> = trigger.event().0.clone();
        info!("Data len: {}", data.len());
        info!("data: {:?}", &data[0..16]);

        let image_handle = world.resource::<Self>().readback_texture.clone();
        let mut images = world.resource_mut::<Assets<Image>>();
        if let Some(image) = images.get_mut(&image_handle) {
            image.data = Some(data);
            let rgba_image: RgbaImage = image_to_rgba(image);
            let _ = rgba_image.save("target/readback_output.png");
        } else {
            warn!("Handle not ready: {:?}", image_handle);
        }
    }
}
impl FromWorld for CustomComputeShader {
    fn from_world(world: &mut World) -> Self {
        let workgroup_size = Self::workgroup_size();
        let size = Extent3d {
            width: workgroup_size.x,
            height: workgroup_size.y,
            depth_or_array_layers: workgroup_size.z,
        };
        let empty_pixel: Vec<u8> = vec![0; 4];
        let mut image = Image::new_fill(
            size,
            TextureDimension::D2,
            &empty_pixel,
            TextureFormat::Rgba8Uint,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING;
        Self {
            texture: world.add_asset(image.clone()),
            readback_texture: world.add_asset(image),
        }
    }
}
