use std::{fmt::Write, num::NonZeroU64};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<wgpu_test::GpuTestInitializer>) {
    vec.push(PIPELINE_CACHE);
    vec.push(PIPELINE_CACHE_GL_PROGRAM_BINARIES);
}

/// We want to test that using a pipeline cache doesn't cause failure
///
/// It would be nice if we could also assert that reusing a pipeline cache would make compilation
/// be faster however, some drivers use a fallback pipeline cache, which makes this inconsistent
/// (both intra- and inter-run).
#[gpu_test]
static PIPELINE_CACHE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::PIPELINE_CACHE),
    )
    .run_async(pipeline_cache_test);

/// Set to a higher value if adding a timing based assertion. This is otherwise fast to compile
const ARRAY_SIZE: u64 = 256;

/// Create a shader which should be slow-ish to compile
fn shader() -> String {
    shader_with_offset(0)
}

/// A shader writing `index + offset` to each element.
fn shader_with_offset(offset: u64) -> String {
    let mut body = String::new();
    for idx in 0..ARRAY_SIZE {
        // "Safety": There will only be a single workgroup, and a single thread in that workgroup
        let value = idx + offset;
        writeln!(body, "    output[{idx}] = {value}u;")
            .expect("`u64::fmt` and `String::write_fmt` are infallible");
    }

    format!(
        r#"
        @group(0) @binding(0)
        var<storage, read_write> output: array<u32>;

        @compute @workgroup_size(1)
        fn main() {{
        {body}
        }}
        "#,
    )
}

async fn pipeline_cache_test(ctx: TestingContext) {
    let shader = shader();
    let sm = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(ARRAY_SIZE * 4),
                },
                count: None,
            }],
        });

    let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_buffer"),
        size: ARRAY_SIZE * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_buffer"),
        size: ARRAY_SIZE * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let first_cache_data;
    {
        let first_cache = unsafe {
            ctx.device
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("pipeline_cache"),
                    data: None,
                    fallback: false,
                })
        };
        let first_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: Some(&pipeline_layout),
                module: &sm,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Some(&first_cache),
            });
        validate_pipeline(&ctx, first_pipeline, &bind_group, &gpu_buffer, &cpu_buffer).await;
        first_cache_data = first_cache.get_data();
    }
    assert!(first_cache_data.is_some());

    let second_cache = unsafe {
        ctx.device
            .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("pipeline_cache"),
                data: first_cache_data.as_deref(),
                fallback: false,
            })
    };
    let first_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &sm,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Some(&second_cache),
        });
    validate_pipeline(&ctx, first_pipeline, &bind_group, &gpu_buffer, &cpu_buffer).await;

    // Ideally, we could assert here that the second compilation was faster than the first
    // However, that doesn't actually work, because drivers have their own internal caches.
    // This does work on my machine if I set `MESA_DISABLE_PIPELINE_CACHE=1`
    // before running the test; but of course that is not a realistic scenario
}

async fn validate_pipeline(
    ctx: &TestingContext,
    pipeline: wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    gpu_buffer: &wgpu::Buffer,
    cpu_buffer: &wgpu::Buffer,
) {
    assert_eq!(
        run_pipeline(ctx, pipeline, bind_group, gpu_buffer, cpu_buffer).await,
        0
    );
}

/// Runs the pipeline and returns the offset its output has from the element indices.
async fn run_pipeline(
    ctx: &TestingContext,
    pipeline: wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    gpu_buffer: &wgpu::Buffer,
    cpu_buffer: &wgpu::Buffer,
) -> u64 {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, Some(bind_group), &[]);

        cpass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(gpu_buffer, 0, cpu_buffer, 0, ARRAY_SIZE * 4);
    ctx.queue.submit([encoder.finish()]);
    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = cpu_buffer.slice(..).get_mapped_range().unwrap();

    let arrays: &[u32] = bytemuck::cast_slice(&data);

    assert_eq!(arrays.len(), ARRAY_SIZE as usize);
    let offset = u64::from(arrays[0]);
    for (idx, value) in arrays.iter().copied().enumerate() {
        assert_eq!(u64::from(value), idx as u64 + offset);
    }
    drop(data);
    cpu_buffer.unmap();
    offset
}

/// On GL the cache holds program binaries. Other backends skip this test.
///
/// - A pipeline created with a cache whose data holds, under its program's key, the binary of
///   another shader computes that shader's values: the program was loaded, not compiled.
/// - A binary the driver rejects (garbage behind a valid entry) is compiled instead, and the
///   fresh binary replaces it in the cache.
/// - Corrupt data (a failed entry checksum) is dropped and the program compiled.
#[gpu_test]
static PIPELINE_CACHE_GL_PROGRAM_BINARIES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::PIPELINE_CACHE),
    )
    .run_async(gl_program_binaries_test);

/// wgpu-core's header before the backend's data (`wgpu_core::pipeline_cache::HEADER_LENGTH`),
/// and the offset of its big-endian data size.
const CORE_HEADER: usize = 64;
const CORE_DATA_SIZE: usize = 48;
/// The GL backend's header (magic, version, adapter key, entry count); see
/// `wgpu_hal::gles::pipeline_cache`.
const GL_HEADER: usize = 8 + 4 + 16 + 4;

/// One entry of GL cache data.
#[derive(Clone, Debug, PartialEq)]
struct GlEntry {
    key: [u8; 16],
    format: u32,
    binary: Vec<u8>,
}

/// The only entry of GL cache data with one program.
fn gl_entry(data: &[u8]) -> GlEntry {
    let hal = &data[CORE_HEADER..];
    assert_eq!(&hal[..8], b"WGPUGLPB");
    assert_eq!(
        u32::from_le_bytes(hal[28..32].try_into().unwrap()),
        1,
        "one entry"
    );
    let entry = &hal[GL_HEADER..];
    let length = u32::from_le_bytes(entry[20..24].try_into().unwrap()) as usize;
    GlEntry {
        key: entry[..16].try_into().unwrap(),
        format: u32::from_le_bytes(entry[16..20].try_into().unwrap()),
        binary: entry[24..24 + length].to_vec(),
    }
}

/// The entry checksum of `wgpu_hal::gles::pipeline_cache` (two FNV-1a lanes).
fn gl_checksum(entry: &GlEntry) -> u64 {
    let mut lanes = [0xcbf2_9ce4_8422_2325u64, 0x6c62_272e_07bb_0142];
    let mut write = |bytes: &[u8]| {
        for lane in &mut lanes {
            for &byte in bytes {
                *lane ^= u64::from(byte);
                *lane = lane.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    };
    write(&16u64.to_le_bytes());
    write(&entry.key);
    write(&entry.format.to_le_bytes());
    write(&(entry.binary.len() as u64).to_le_bytes());
    write(&entry.binary);
    lanes[0] ^ lanes[1].rotate_left(32)
}

/// `data` (one program) with its entry replaced by `entry`, checksummed and sized correctly.
fn with_gl_entry(data: &[u8], entry: &GlEntry) -> Vec<u8> {
    let mut out = data[..CORE_HEADER + GL_HEADER].to_vec();
    out.extend_from_slice(&entry.key);
    out.extend_from_slice(&entry.format.to_le_bytes());
    out.extend_from_slice(&(entry.binary.len() as u32).to_le_bytes());
    out.extend_from_slice(&entry.binary);
    out.extend_from_slice(&gl_checksum(entry).to_le_bytes());
    let size = (out.len() - CORE_HEADER) as u64;
    out[CORE_DATA_SIZE..CORE_DATA_SIZE + 8].copy_from_slice(&size.to_be_bytes());
    out
}

async fn gl_program_binaries_test(ctx: TestingContext) {
    if unsafe { ctx.device.as_hal::<wgpu::hal::api::Gles>() }.is_none() {
        return;
    }
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(ARRAY_SIZE * 4),
                },
                count: None,
            }],
        });
    let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_buffer"),
        size: ARRAY_SIZE * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_buffer"),
        size: ARRAY_SIZE * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_buffer.as_entire_binding(),
        }],
    });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let cache = |data: Option<&[u8]>| unsafe {
        ctx.device
            .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("pipeline_cache"),
                data,
                fallback: true,
            })
    };
    // A new module for every pipeline: GL reuses a linked program for the same module and
    // layout, which would never reach the cache's data.
    let run = |offset: u64, cache: &wgpu::PipelineCache| {
        let source = shader_with_offset(offset);
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Some(cache),
            });
        run_pipeline(&ctx, pipeline, &bind_group, &gpu_buffer, &cpu_buffer)
    };

    // The binaries of two shaders, one cache each.
    let plain = cache(None);
    assert_eq!(run(0, &plain).await, 0);
    let plain_data = plain.get_data().expect("GL caches have data");
    let plain_entry = gl_entry(&plain_data);
    let shifted = cache(None);
    assert_eq!(run(1000, &shifted).await, 1000);
    let shifted_entry = gl_entry(&shifted.get_data().unwrap());
    assert_ne!(plain_entry.key, shifted_entry.key);

    // The plain shader's key with the shifted shader's binary: the program is loaded from the
    // data, so it computes the shifted values.
    let swapped = with_gl_entry(
        &plain_data,
        &GlEntry {
            key: plain_entry.key,
            ..shifted_entry.clone()
        },
    );
    assert_eq!(
        run(0, &cache(Some(&swapped))).await,
        1000,
        "the pipeline used the cached binary instead of compiling its shader"
    );

    // Valid data the driver rejects: compiled instead, and the fresh binary replaces it.
    let rejected = cache(Some(&with_gl_entry(
        &plain_data,
        &GlEntry {
            binary: vec![0xab; 64],
            ..plain_entry.clone()
        },
    )));
    assert_eq!(run(0, &rejected).await, 0, "compiled after the rejection");
    let replaced = gl_entry(&rejected.get_data().unwrap());
    assert_eq!(replaced.key, plain_entry.key);
    assert_ne!(
        replaced.binary,
        vec![0xab; 64],
        "the rejected binary is gone"
    );

    // Corrupt data: the entry fails its checksum, is dropped, and the program is compiled.
    let mut corrupt = plain_data.clone();
    let last = corrupt.len() - 9;
    corrupt[last] ^= 0xff;
    let corrupt = cache(Some(&corrupt));
    assert_eq!(run(0, &corrupt).await, 0);
    assert_eq!(gl_entry(&corrupt.get_data().unwrap()).key, plain_entry.key);

    // A program an earlier pipeline linked goes into a cache passed later.
    let warm = cache(None);
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(shader_with_offset(7).into()),
        });
    let pipeline = |cache: Option<&wgpu::PipelineCache>| {
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache,
            })
    };
    let first = pipeline(None);
    let second = pipeline(Some(&warm));
    assert_eq!(
        run_pipeline(&ctx, second, &bind_group, &gpu_buffer, &cpu_buffer).await,
        7
    );
    drop(first);
    assert!(
        warm.get_data()
            .is_some_and(|data| !gl_entry(&data).binary.is_empty()),
        "the shared program's binary was stored"
    );
}
