from pathlib import Path

import h5py


file_path = Path(r"C:\HACKATHONS\SIH 2026\test_142.h5")
output_path = Path(__file__).resolve().with_name("output_44.txt")

if not file_path.is_file():
    raise FileNotFoundError(f"HDF5 file not found: {file_path}")

with h5py.File(file_path, "r") as file:
    required_datasets = {"data", "labels"}
    missing_datasets = required_datasets.difference(file.keys())
    if missing_datasets:
        raise KeyError(f"Missing datasets: {sorted(missing_datasets)}")

    data = file["data"][:]
    labels = file["labels"][:]

    if len(data) != len(labels):
        raise ValueError(
            f"Data/label length mismatch: {len(data)} data rows, "
            f"{len(labels)} labels"
        )

    output_lines = []
    output_lines.append("dataset_names: " + ", ".join(file.keys()))
    output_lines.append(f"data_shape: {data.shape}")
    output_lines.append(f"data_dtype: {data.dtype}")
    output_lines.append(f"labels_shape: {labels.shape}")
    output_lines.append(f"labels_dtype: {labels.dtype}")
    output_lines.append(f"record_count: {len(data)}")
    output_lines.append("records:")

    for row_number, (row, label) in enumerate(zip(data, labels.ravel()), start=1):
        output_lines.append(
            f"record_{row_number}: data={row.tolist()}, label={int(label)}"
        )

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

# No console dump of every row; the structured data is saved in output_0.txt.
print(f"Structured output saved to: {output_path}")