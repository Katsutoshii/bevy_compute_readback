// Creates a texture all with the same color.
@group(0) @binding(0) var texture: texture_storage_2d<rgba8uint, write>;

// Populate the texture with whatever function you like.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texture_xy = vec2<i32>(global_id.xy);
    let color = vec4<u32>(32 + global_id.x, 64 + global_id.y, 96 + global_id.z, 128);
    textureStore(texture, texture_xy, color);
}
