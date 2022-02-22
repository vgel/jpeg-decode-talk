from __future__ import annotations

from dataclasses import dataclass
import math
import sys

import numpy as np
import PIL.Image


# Read image bytes

with open(sys.argv[1], "rb") as f:
    image_data = f.read()

# Segment definitions


@dataclass
class Segment:
    # these aren't all the kinds, just the ones we care about
    KINDS = {
        0xC0: "StartOfFrameBaseline",
        0xC4: "DefineHuffmanTable",
        0xD8: "StartOfImage",
        0xD9: "EndOfImage",
        0xDA: "StartOfScan",
        0xDB: "DefineQuantTable",
        0xE0: "AppSpecific",
        0xFE: "Comment",
    }
    OTHER = "Other"  # unknown kind byte

    kind: str
    length: int
    data: bytes

    def __repr__(self) -> str:
        if len(self.data) <= 20:
            data = repr(self.data)
        else:
            data = repr(self.data[:17]) + "..."
        return f"Segment(kind={self.kind!r}, length={self.length}, data={data})"


def read_segment(data: bytes) -> tuple[Segment, bytes]:
    """Reads a segment, returning it and any bytes that follow it"""

    marker_byte, kind_byte = data[0], data[1]
    assert marker_byte == 0xFF
    has_no_data = kind_byte in range(0xD0, 0xD9 + 1)
    kind = Segment.KINDS.get(kind_byte, Segment.OTHER)

    if has_no_data:
        remainder = data[2:]
        return Segment(kind, length=0, data=bytes()), remainder
    else:
        length = (data[2] << 8) + data[3] - 2
        payload = data[4 : 4 + length]
        remainder = data[4 + length :]

        if kind == "StartOfScan":
            # read through huffman data for next segment
            while len(remainder):
                if remainder[:2] == bytes([0xFF, 0x00]):  # escape sequence
                    payload += bytes([0xFF])
                    remainder = remainder[2:]
                elif remainder[0] == 0xFF:  # a new segment header!
                    break
                else:  # a regular byte
                    payload += remainder[:1]
                    remainder = remainder[1:]

        return Segment(kind, length, payload), remainder


def read_all_segments(data: bytes) -> dict[str, list[Segment]]:
    """Reads all the segments in data"""

    result: dict[str, list[Segment]] = {}

    while len(data):
        segment, data = read_segment(data)

        if segment.kind not in result:
            result[segment.kind] = [segment]
        else:
            result[segment.kind].append(segment)

    return result


image_segments = read_all_segments(image_data)

# Parse StartOfFrame


@dataclass
class Component:
    id: int
    sampling_res: int
    quantization_table: int


@dataclass
class FrameInfo:
    width: int
    height: int
    components: dict[int, Component]


def read_start_of_frame(data: bytes) -> FrameInfo:
    precision = data[0]
    assert precision == 8

    height = (data[1] << 8) + data[2]
    width = (data[3] << 8) + data[4]

    n_components = data[5]
    assert n_components == 3  # JFIF

    component_data = data[6:]
    components = [
        Component(
            id=component_data[i * 3],
            sampling_res=component_data[i * 3 + 1],
            quantization_table=component_data[i * 3 + 2],
        )
        for i in range(3)
    ]

    return FrameInfo(width, height, {c.id: c for c in components})


image_frameinfo = read_start_of_frame(image_segments["StartOfFrameBaseline"][0].data)

# Now we can print some information about the image!

print("Image info:")
print(f" Dimensions: ({image_frameinfo.width}, {image_frameinfo.height})")
for c in image_frameinfo.components.values():
    print(f" Component {c.id}:")
    print(f"  Sampling Resolution: {c.sampling_res}")
    print(f"  Quantization Table:  {c.quantization_table}")
print()

# Parse Quantization tables

image_quantization_table_segments = image_segments["DefineQuantTable"]
print(f"Found {len(image_quantization_table_segments)} quantization table segments")


@dataclass
class QuantizationTable:
    id: int
    values: list[int]


def read_quantization_table(data: bytes) -> QuantizationTable:
    header = data[0]
    precision = header >> 4
    assert precision == 0, f"only 8-bit quantization precision is supported"
    assert len(data) == 65, f"only 1 quantization table per segment is supported"

    id = header & 0x0F
    values = list(data[1:])
    return QuantizationTable(id, values)


image_quantization_tables: dict[int, QuantizationTable] = {}
for qt_seg in image_quantization_table_segments:
    qt = read_quantization_table(qt_seg.data)
    image_quantization_tables[qt.id] = qt

print("Quantization Tables:")
for qt_table in image_quantization_tables.values():
    print(f" Table {qt_table.id}:")
    for i in range(0, 64, 8):
        print("  " + " ".join(str(i).rjust(3) for i in qt_table.values[i : i + 8]))
print()

# Parse Huffman tables

image_huffman_table_segments = image_segments["DefineHuffmanTable"]
print(f"Found {len(image_huffman_table_segments)} Huffman table segments")


@dataclass(eq=True, frozen=True)
class HuffmanTableId:
    kind: str
    id: int


@dataclass
class HuffmanTable:
    id: HuffmanTableId
    code_to_value: dict[int, int]

    def __contains__(self, key) -> bool:
        return key in self.code_to_value

    def __getitem__(self, key) -> int:
        return self.code_to_value[key]


def read_huffman_table(data: bytes) -> HuffmanTable:
    kind = "dc" if data[0] & 0xF0 == 0 else "ac"
    id = data[0] & 0x0F

    codes_per_code_length = data[1:17]
    values = list(data[17:])

    code = 0
    code_to_value: dict[int, int] = {}
    for code_len in range(1, 17):
        for _ in range(codes_per_code_length[code_len - 1]):
            code_to_value[code] = values[len(code_to_value)]
            code += 1
        code *= 2  # add zero to the end of code

    return HuffmanTable(HuffmanTableId(kind, id), code_to_value)


image_huffman_tables: dict[HuffmanTableId, HuffmanTable] = {}
for ht_segment in image_segments["DefineHuffmanTable"]:
    ht = read_huffman_table(ht_segment.data)
    image_huffman_tables[ht.id] = ht

print("Huffman Tables:")
for h_table in image_huffman_tables.values():
    print(f" {h_table.id!r}:")
    for code, value in h_table.code_to_value.items():
        print(f"  {bin(code)[2:]}: {value}")
print()

# Parse StartOfScan


@dataclass
class ComponentInfo:
    id: int
    dc_table: HuffmanTableId
    ac_table: HuffmanTableId


@dataclass
class ScanInfo:
    component_huffman_tables: dict[int, ComponentInfo]
    # we won't be using these fields, but we'll parse them out anyways because
    # they're easy
    spectral_selection_start: int
    spectral_selection_end: int
    successive_approximation: int


def read_start_of_scan(data: bytes) -> ScanInfo:
    component_count = data[0]
    assert component_count == 3, "Only 3-component JFIF is supported"

    component_huffman_tables: dict[int, ComponentInfo] = {}
    for i in range(3):
        component_id = data[1 + i * 2 + 0]
        component_tb = data[1 + i * 2 + 1]

        component_huffman_tables[component_id] = ComponentInfo(
            id=component_id,
            dc_table=HuffmanTableId("dc", (component_tb & 0xF0) >> 4),
            ac_table=HuffmanTableId("ac", component_tb & 0x0F),
        )

    # there are additional fields in this header, but we don't need them
    spectral_selection_start = data[7]
    spectral_selection_end = data[8]
    successive_approximation = data[9]

    return ScanInfo(
        component_huffman_tables,
        spectral_selection_start,
        spectral_selection_end,
        successive_approximation,
    )


image_start_of_scan_segment = image_segments["StartOfScan"][0]
image_scan_info = read_start_of_scan(
    image_start_of_scan_segment.data[: image_start_of_scan_segment.length]
)
print("Image Scan Info:")
for c_ht_info in image_scan_info.component_huffman_tables.values():
    print(f" Component {c_ht_info.id}:")
    print(f"  DC table: {c_ht_info.dc_table!r}")
    print(f"  AC table: {c_ht_info.ac_table!r}")
print(f" Spectral selection start: {image_scan_info.spectral_selection_start}")
print(f" Spectral selection end: {image_scan_info.spectral_selection_end}")
print(f" Successive approximation: {image_scan_info.successive_approximation}")
print()

# Define a ImageMetadata structure to hold all the metadata we've parsed out


@dataclass
class ComponentMetadata:
    id: int
    sampling_res: int
    quantization_table: QuantizationTable
    huffman_dc_table: HuffmanTable
    huffman_ac_table: HuffmanTable


@dataclass
class ImageMetadata:
    width: int
    height: int
    quantization_tables: dict[int, QuantizationTable]
    huffman_tables: dict[HuffmanTableId, HuffmanTable]
    components: dict[int, ComponentMetadata]


image_metadata = ImageMetadata(
    width=image_frameinfo.width,
    height=image_frameinfo.height,
    quantization_tables=image_quantization_tables,
    huffman_tables=image_huffman_tables,
    components={
        id: ComponentMetadata(
            id=id,
            sampling_res=image_frameinfo.components[id].sampling_res,
            quantization_table=image_quantization_tables[
                image_frameinfo.components[id].quantization_table
            ],
            huffman_dc_table=image_huffman_tables[
                image_scan_info.component_huffman_tables[id].dc_table
            ],
            huffman_ac_table=image_huffman_tables[
                image_scan_info.component_huffman_tables[id].ac_table
            ],
        )
        for id in image_frameinfo.components.keys()
    },
)

# Bitstream decompression


@dataclass
class Bitstream:
    cursor: int
    bits: list[int]

    @classmethod
    def from_bytes(cls, data: bytes) -> Bitstream:
        # very ineffecient but good enough for the small images we're using
        bits = [int(c) for c in "".join(bin(i)[2:].rjust(8, "0") for i in data)]
        return cls(0, bits)

    def read(self, count: int) -> list[int]:
        if count > len(self):
            raise ValueError(f"read: overrun ({count} > {len(self)})")

        prev_cursor = self.cursor
        self.cursor += count
        return self.bits[prev_cursor : self.cursor]

    def read_unsigned(self, count: int) -> int:
        i = 0
        for bit in self.read(count):
            i = (i << 1) + bit
        return i

    def read_signed(self, count: int) -> int:
        if count == 0:
            return 0
        elif self.bits[self.cursor] == 1:  # positive
            return self.read_unsigned(count)
        else:  # negative offset
            return self.read_unsigned(count) - (2 ** count - 1)

    def read_huffman(self, table: HuffmanTable) -> int:
        for code_len in range(1, 17):
            code = self.read_unsigned(code_len)
            if code in table:
                return table[code]
            else:
                self.cursor -= code_len  # reset for next iteration

        raise RuntimeError("unterminated huffman sequence")

    def __len__(self) -> int:
        return len(self.bits) - self.cursor


# Sub-block decompression


def read_subblock(
    bits: Bitstream,
    component: ComponentMetadata,
    prev_dc: int,
) -> tuple[int, np.ndarray]:
    sub_block = np.zeros(64)

    # read the dc offset, if any
    n_dc_bits = bits.read_huffman(component.huffman_dc_table)
    dc = prev_dc + bits.read_signed(n_dc_bits)
    sub_block[0] = dc

    ac_idx = 1  # dc is 0
    while ac_idx < 64:
        header = bits.read_huffman(component.huffman_ac_table)
        if header == 0:
            break  # marker to skip rest of block

        n_leading_zeros = (header & 0xF0) >> 4
        ac_idx += n_leading_zeros
        sub_block[ac_idx] = bits.read_signed(header & 0x0F)
        ac_idx += 1

    return (dc, sub_block.reshape((8, 8)))


def read_subblocks(
    data: bytes,
    image: ImageMetadata,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    bits = Bitstream.from_bytes(data)

    prev_luma_dc, prev_chcb_dc, prev_chcr_dc = 0, 0, 0
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    while len(bits) >= 8:  # ignore any padding bits at end
        # order of components is 1-3-2
        prev_luma_dc, luma = read_subblock(bits, image.components[1], prev_luma_dc)
        prev_chcb_dc, chcb = read_subblock(bits, image.components[3], prev_chcb_dc)
        prev_chcr_dc, chcr = read_subblock(bits, image.components[2], prev_chcr_dc)
        blocks.append((luma, chcr, chcb))

    return blocks


subblock_data = image_start_of_scan_segment.data[image_start_of_scan_segment.length :]
blocks = read_subblocks(subblock_data, image_metadata)

# Dequantize and unzigzag sub-blocks

# fmt: off
UNZIGZAG = [
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
]
# fmt: on


def dequantize_subblock(
    subblock: np.ndarray, component: ComponentMetadata
) -> np.ndarray:
    unzig = subblock.flatten()[UNZIGZAG]
    dequant = unzig * component.quantization_table.values
    return dequant.reshape((8, 8))


def dequantize_subblocks(
    subblocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]], image: ImageMetadata
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (
            dequantize_subblock(luma, image.components[1]),
            dequantize_subblock(chcr, image.components[2]),
            dequantize_subblock(chcb, image.components[3]),
        )
        for (luma, chcr, chcb) in subblocks
    ]


dequantized_blocks = dequantize_subblocks(blocks, image_metadata)

# IDCT transform


def f_dct(row: int, col: int) -> float:
    norm = 1 / math.sqrt(2) if row == 0 else 1
    cos = math.cos((2 * col + 1) * row * math.pi / 16)
    return norm * cos / 2


DCT = np.array([[f_dct(row, col) for col in range(8)] for row in range(8)])


def idct_subblock(subblock: np.ndarray) -> np.ndarray:
    return ((DCT.T @ subblock.T @ DCT) + 128).round().clip(0, 255).astype(int)


# Color space conversion


def convert_rgb_subblocks(
    dequantized_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rgb_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for luma, chcr, chcb in dequantized_blocks:
        y = idct_subblock(luma)
        cr = idct_subblock(chcr)
        cb = idct_subblock(chcb)

        r = y + 1.402 * (cr - 128)
        g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
        b = y + 1.772 * (cb - 128)

        r = r.clip(0, 255).astype(int)
        g = g.clip(0, 255).astype(int)
        b = b.clip(0, 255).astype(int)

        rgb_blocks.append((r, g, b))
    return rgb_blocks


rgb_blocks = convert_rgb_subblocks(dequantized_blocks)

# Tile blocks into a single PIL image


def make_image(
    rgb_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> PIL.Image.Image:
    buffer = bytearray(image_metadata.width * image_metadata.height * 3)
    for y in range(image_metadata.height):
        for x in range(image_metadata.width):
            block_x = x // 8
            block_y = y // 8
            block_idx = block_x + block_y * (image_metadata.width // 8)

            r, g, b = rgb_blocks[block_idx]

            buffer[(x + y * image_metadata.width) * 3 + 0] = r[x % 8, y % 8]
            buffer[(x + y * image_metadata.width) * 3 + 1] = g[x % 8, y % 8]
            buffer[(x + y * image_metadata.width) * 3 + 2] = b[x % 8, y % 8]

    return PIL.Image.frombytes(
        "RGB", (image_metadata.width, image_metadata.height), bytes(buffer)
    )


# Extremely thorough test suite

decoded = make_image(rgb_blocks)
assert decoded.tobytes() == PIL.Image.open("subject.png").tobytes()

if len(sys.argv) == 3:
    decoded.save(sys.argv[2])
