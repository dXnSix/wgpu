//! Program binaries behind the GL pipeline cache.
//!
//! A [`super::PipelineCache`] maps each linked program's key (a hash of its GLSL sources and the
//! adapter's `GL_RENDERER` and `GL_VERSION`) to the program binary `glGetProgramBinary` returned,
//! with its format. A later run loads it with `glProgramBinary` instead of compiling, and compiles
//! when the driver rejects it.
//!
//! The data is validated before use, so stale or corrupt data only costs a compile:
//!
//! ```text
//! magic "WGPUGLPB" | version u32 | adapter key [u8; 16] | count u32
//! count × ( program key [u8; 16] | format u32 | length u32 | binary [u8; length] | checksum u64 )
//! ```
//!
//! Integers are little-endian. Data with another magic, version or adapter key yields an empty
//! cache; an entry whose checksum does not match its key, format and binary is dropped, and a
//! truncated entry ends the list.

use alloc::{collections::BTreeMap, vec::Vec};

const MAGIC: [u8; 8] = *b"WGPUGLPB";
const VERSION: u32 = 1;

/// A stable 128-bit hash (two 64-bit FNV-1a lanes with different offsets); never `std`'s
/// randomly seeded one, since keys persist across runs.
#[derive(Clone)]
pub(super) struct StableHasher {
    lanes: [u64; 2],
}

impl StableHasher {
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub(super) fn new() -> Self {
        Self {
            lanes: [0xcbf2_9ce4_8422_2325, 0x6c62_272e_07bb_0142],
        }
    }

    pub(super) fn write(&mut self, bytes: &[u8]) {
        for lane in &mut self.lanes {
            for &byte in bytes {
                *lane ^= u64::from(byte);
                *lane = lane.wrapping_mul(Self::PRIME);
            }
        }
    }

    /// Writes `bytes` with its length first, so `["ab", "c"]` and `["a", "bc"]` differ.
    pub(super) fn write_part(&mut self, bytes: &[u8]) {
        self.write(&(bytes.len() as u64).to_le_bytes());
        self.write(bytes);
    }

    pub(super) fn finish(&self) -> [u8; 16] {
        let mut out = [0; 16];
        out[..8].copy_from_slice(&self.lanes[0].to_le_bytes());
        out[8..].copy_from_slice(&self.lanes[1].to_le_bytes());
        out
    }

    fn finish_u64(&self) -> u64 {
        self.lanes[0] ^ self.lanes[1].rotate_left(32)
    }
}

/// The adapter part of every key: `GL_RENDERER` and `GL_VERSION`.
pub(super) fn adapter_key(renderer: &str, version: &str) -> [u8; 16] {
    let mut hasher = StableHasher::new();
    hasher.write_part(b"adapter");
    hasher.write_part(renderer.as_bytes());
    hasher.write_part(version.as_bytes());
    hasher.finish()
}

/// A program's key: its stages' GLSL sources, in order, on the adapter.
pub(super) fn program_key<'a>(
    adapter: &[u8; 16],
    sources: impl IntoIterator<Item = &'a str>,
) -> [u8; 16] {
    let mut hasher = StableHasher::new();
    hasher.write_part(b"program");
    hasher.write_part(adapter);
    for source in sources {
        hasher.write_part(source.as_bytes());
    }
    hasher.finish()
}

/// A stored program binary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Binary {
    pub format: u32,
    pub data: Vec<u8>,
}

fn checksum(key: &[u8; 16], binary: &Binary) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.write_part(key);
    hasher.write(&binary.format.to_le_bytes());
    hasher.write_part(&binary.data);
    hasher.finish_u64()
}

/// The cache data for `entries` on the adapter `adapter`.
pub(super) fn encode(adapter: &[u8; 16], entries: &BTreeMap<[u8; 16], Binary>) -> Vec<u8> {
    let size: usize = entries.values().map(|b| 16 + 8 + b.data.len() + 8).sum();
    let mut out = Vec::with_capacity(8 + 4 + 16 + 4 + size);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(adapter);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (key, binary) in entries {
        out.extend_from_slice(key);
        out.extend_from_slice(&binary.format.to_le_bytes());
        out.extend_from_slice(&(binary.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&binary.data);
        out.extend_from_slice(&checksum(key, binary).to_le_bytes());
    }
    out
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    fn u32(&mut self) -> Option<u32> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Option<u64> {
        self.array().map(u64::from_le_bytes)
    }
}

/// The valid entries of `data` for the adapter `adapter` (see the module docs).
pub(super) fn decode(adapter: &[u8; 16], data: &[u8]) -> BTreeMap<[u8; 16], Binary> {
    let mut entries = BTreeMap::new();
    let mut reader = Reader(data);
    let header_ok = reader.array::<8>() == Some(MAGIC)
        && reader.u32() == Some(VERSION)
        && reader.array::<16>().as_ref() == Some(adapter);
    let Some(count) = header_ok.then(|| reader.u32()).flatten() else {
        return entries;
    };
    for _ in 0..count {
        let Some((key, binary, sum)) = (|| {
            let key = reader.array::<16>()?;
            let format = reader.u32()?;
            let length = reader.u32()? as usize;
            let data = reader.take(length)?.to_vec();
            let sum = reader.u64()?;
            Some((key, Binary { format, data }, sum))
        })() else {
            break;
        };
        if checksum(&key, &binary) == sum {
            entries.insert(key, binary);
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> BTreeMap<[u8; 16], Binary> {
        let adapter = adapter_key("llvmpipe", "OpenGL ES 3.2 Mesa");
        let mut entries = BTreeMap::new();
        entries.insert(
            program_key(&adapter, ["vertex", "fragment"]),
            Binary {
                format: 0x8741,
                data: (0..200u8).collect(),
            },
        );
        entries.insert(
            program_key(&adapter, ["compute"]),
            Binary {
                format: 0x8741,
                data: alloc::vec![7; 3],
            },
        );
        entries
    }

    #[test]
    fn keys_cover_sources_their_order_and_the_adapter() {
        let mali = adapter_key("Mali-G52", "OpenGL ES 3.2 v1.r26");
        let other_driver = adapter_key("Mali-G52", "OpenGL ES 3.2 v1.r32");
        assert_ne!(
            mali, other_driver,
            "a driver update invalidates the binaries"
        );
        let key = program_key(&mali, ["a", "b"]);
        assert_eq!(key, program_key(&mali, ["a", "b"]), "stable");
        assert_ne!(key, program_key(&mali, ["b", "a"]));
        assert_ne!(key, program_key(&mali, ["ab", ""]));
        assert_ne!(key, program_key(&other_driver, ["a", "b"]));
        // Fixed across runs and builds: a known value.
        let mut hasher = StableHasher::new();
        hasher.write(b"a");
        assert_eq!(
            &hasher.finish()[..8],
            &0xaf63_dc4c_8601_ec8c_u64.to_le_bytes()
        );
    }

    #[test]
    fn data_round_trips() {
        let adapter = adapter_key("llvmpipe", "OpenGL ES 3.2 Mesa");
        let entries = entries();
        let data = encode(&adapter, &entries);
        assert_eq!(decode(&adapter, &data), entries);
        assert!(decode(&adapter, &encode(&adapter, &BTreeMap::new())).is_empty());
    }

    #[test]
    fn stale_or_corrupt_data_falls_back() {
        let adapter = adapter_key("llvmpipe", "OpenGL ES 3.2 Mesa");
        let entries = entries();
        let data = encode(&adapter, &entries);
        let other = adapter_key("llvmpipe", "OpenGL ES 3.2 Mesa 25");
        assert!(
            decode(&other, &data).is_empty(),
            "another adapter or driver"
        );
        let mut bad_magic = data.clone();
        bad_magic[0] ^= 1;
        assert!(decode(&adapter, &bad_magic).is_empty());
        let mut bad_version = data.clone();
        bad_version[8] = 9;
        assert!(decode(&adapter, &bad_version).is_empty());
        assert!(decode(&adapter, &data[..10]).is_empty(), "truncated header");
        assert!(decode(&adapter, &[]).is_empty());

        // A flipped byte in the first entry's binary drops that entry only.
        let mut flipped = data.clone();
        flipped[8 + 4 + 16 + 4 + 16 + 8 + 5] ^= 0xff;
        let decoded = decode(&adapter, &flipped);
        assert_eq!(decoded.len(), 1);
        let (key, binary) = decoded.iter().next().unwrap();
        assert_eq!(entries.get(key), Some(binary));

        // A truncated entry ends the list; earlier entries stay.
        let decoded = decode(&adapter, &data[..data.len() - 4]);
        assert_eq!(decoded.len(), 1);
    }
}
