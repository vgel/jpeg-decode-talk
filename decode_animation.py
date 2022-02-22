import pathlib
import os

from decode import *

REZIGZAG = [UNZIGZAG.index(i) for i in range(64)]

def truncate_quant_table(qt: QuantizationTable, n_comps: int) -> QuantizationTable:
    coeffs = list(qt.values)
    for idx in REZIGZAG[n_comps:]:
        coeffs[idx] = 0
    return QuantizationTable(qt.id, coeffs)


def truncate_comp_meta(cm: ComponentMetadata, n_comps: int) -> ComponentMetadata:
    return ComponentMetadata(
        cm.id,
        cm.sampling_res,
        truncate_quant_table(cm.quantization_table, n_comps),
        cm.huffman_dc_table,
        cm.huffman_ac_table,
    )

output_dir = pathlib.Path("anim_frames")

def frame_fn(i: int) -> str:
    return str(output_dir / f"frame_{str(i).zfill(2)}.png")

if not output_dir.exists():
    output_dir.mkdir()

    for i in range(1, 65):
        truncated_image_meta = ImageMetadata(
            image_metadata.width,
            image_metadata.height,
            {
                k: truncate_quant_table(qt, i)
                for k, qt in image_metadata.quantization_tables.items()
            },
            image_metadata.huffman_tables,
            {
                k: truncate_comp_meta(comp, i)
                for k, comp in image_metadata.components.items()
            },
        )

        dequantized_blocks = dequantize_subblocks(blocks, truncated_image_meta)
        rgb_blocks = convert_rgb_subblocks(dequantized_blocks)
        decoded = make_image(rgb_blocks)
        decoded.save(frame_fn(i))
    
    for i in range(1, 65):
        os.system(" ".join((
            f"convert -font helvetica -fill red -pointsize 30",
            f"-draw \"text 20,30 '{i}'\"",
            f"{frame_fn(i)} {frame_fn(i)}",
        )))

def fns(*args: int) -> str:
    return " ".join(frame_fn(i) for i in args)

os.system(" ".join((
    f"convert -loop 0",
    f"-delay 50 {fns(*range(1, 5))}",
    f"-delay 20 {fns(*range(5, 17))}",
    f"-delay 2 {fns(*range(17, 64))}",
    f"anim.gif",
)))