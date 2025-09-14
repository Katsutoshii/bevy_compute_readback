// Compute shader that populates a texture.
@group(0) @binding(0) var texture: texture_storage_2d<rgba32float, write>;

@group(0) @binding(1) var<uniform> color: vec4<f32>;

// Function to return a color for a given (x, y) point.
fn f(xy: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(xy.x + 0.25, xy.y + 0.5, 0.75, 0.5);
}

// Writes the function value to each pixel of the texture.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texture_xy = vec2<i32>(global_id.xy);
    let g = vec3<f32>(global_id) / 64.0;
    textureStore(texture, texture_xy, color * f(g.xy));
}
