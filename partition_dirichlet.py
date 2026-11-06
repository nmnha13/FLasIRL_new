#!/usr/bin/env python3
"""
Tạo partition Non-IID cho tập dữ liệu MNIST bằng phân phối Dirichlet (Dirichlet Distribution).

File này phân chia dữ liệu MNIST (60.000 mẫu train) cho K client (mặc định 25 client)
dựa trên phân phối Dirichlet với tham số tập trung alpha (Dirichlet concentration parameter).
- alpha càng nhỏ (ví dụ 0.1, 0.5): độ Non-IID càng cao (dữ liệu các client rất lệch class).
- alpha càng lớn (ví dụ 10, 100): phân phối càng tiến gần về IID (đồng đều giữa các class).

Kết quả partition được lưu trực tiếp vào thư mục PARTITION_DIR (PROJECT_ROOT/partitions)
với định dạng file .pt hoàn toàn tương thích với pipeline huấn luyện Federated Learning.
Biểu đồ phân bố class của các client được lưu vào CHART_DIR (PROJECT_ROOT/charts).
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets


def get_default_project_root() -> Path:
    """Tự động xác định PROJECT_ROOT phù hợp cho Google Colab hoặc môi trường local."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        project_root = Path('/content/drive/MyDrive/FLasIRL(new)')
    except ImportError:
        project_root = Path.cwd() / 'FLasIRL(new)' if not Path('/content').exists() else Path('/content/FLasIRL(new)')

    if not project_root.parent.exists():
        project_root = Path.cwd()
    return project_root


def generate_dirichlet_partition(
    train_labels: torch.Tensor,
    num_clients: int = 25,
    alpha: float = 0.5,
    seed: int = 20261106,
    min_samples: int = 10,
    max_retries: int = 100,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Phân chia chỉ số tập dữ liệu theo phân phối Dirichlet trên từng class (Dirichlet-Multinomial).

    Args:
        train_labels: Tensor chứa nhãn của toàn bộ tập train (MNIST có 60.000 mẫu, 10 class).
        num_clients: Số lượng client (mặc định 25).
        alpha: Tham số Dirichlet concentration.
        seed: Hạt giống ngẫu nhiên để đảm bảo tính tái lập (reproducibility).
        min_samples: Số lượng mẫu tối thiểu cho mỗi client để tránh client không có dữ liệu.
        max_retries: Số lần thử lại tối đa nếu có client nhận ít hơn min_samples.

    Returns:
        client_indices: Danh sách gồm num_clients Tensor 1D chứa các chỉ số mẫu cho từng client.
        client_class_counts: Tensor kích thước (num_clients, num_classes) lưu số lượng mẫu từng class.
    """
    labels_np = train_labels.cpu().numpy()
    num_samples = len(labels_np)
    classes = np.unique(labels_np)
    num_classes = len(classes)

    current_seed = seed
    for attempt in range(max_retries):
        rng = np.random.default_rng(current_seed)
        client_sample_indices = [[] for _ in range(num_clients)]

        # Phân phối từng class độc lập theo Dirichlet
        for c in classes:
            idx_c = np.where(labels_np == c)[0]
            rng.shuffle(idx_c)

            # Lấy vector tỷ lệ từ phân phối Dirichlet(alpha, ..., alpha) cho K client
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))

            # Sử dụng phân phối Multinomial để chia đúng số lượng mẫu của class c mà không bị lệch làm tròn
            counts = rng.multinomial(len(idx_c), proportions)

            start = 0
            for client_id, count in enumerate(counts):
                if count > 0:
                    client_sample_indices[client_id].extend(idx_c[start : start + count])
                    start += count

        # Kiểm tra xem có client nào bị quá ít mẫu không
        client_sizes = [len(indices) for indices in client_sample_indices]
        if min(client_sizes) >= min_samples:
            # Thành công, xáo trộn chỉ số nội bộ của từng client
            client_indices = []
            client_class_counts = torch.zeros((num_clients, num_classes), dtype=torch.long)

            for client_id, indices in enumerate(client_sample_indices):
                indices_arr = np.array(indices, dtype=np.int64)
                rng.shuffle(indices_arr)
                t_indices = torch.tensor(indices_arr, dtype=torch.long)
                client_indices.append(t_indices)
                client_class_counts[client_id] = torch.bincount(train_labels[t_indices], minlength=num_classes)

            # Kiểm tra tính toàn vẹn (integrity check)
            assert sum(len(x) for x in client_indices) == num_samples, "Tổng số mẫu phân chia không khớp!"
            assert len(torch.cat(client_indices).unique()) == num_samples, "Có mẫu bị trùng lặp giữa các client!"

            return client_indices, client_class_counts

        current_seed += 1

    raise RuntimeError(
        f"Không thể tạo partition Dirichlet với alpha={alpha} và min_samples={min_samples} "
        f"sau {max_retries} lần thử. Hãy giảm min_samples hoặc tăng alpha."
    )


def plot_and_save_distribution(
    client_class_counts: torch.Tensor,
    num_clients: int,
    alpha: float,
    seed: int,
    chart_path: Path,
) -> None:
    """Vẽ và lưu biểu đồ cột phân bố class của các client."""
    num_classes = client_class_counts.shape[1]
    client_sizes = client_class_counts.sum(dim=1).cpu().numpy()

    plt.figure(figsize=(14, 6))
    bottom = np.zeros(num_clients)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for cls in range(num_classes):
        values = client_class_counts[:, cls].cpu().numpy()
        plt.bar(
            np.arange(num_clients),
            values,
            bottom=bottom,
            label=f"Class {cls}",
            color=colors[cls],
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += values

    plt.xlabel("Client ID", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title(
        f"Non-IID Dirichlet Class Distribution (K={num_clients}, $\\alpha$={alpha}, Seed={seed})\n"
        f"Min Samples: {client_sizes.min()}, Max: {client_sizes.max()}, Mean: {client_sizes.mean():.1f}",
        fontsize=13,
    )
    plt.xticks(np.arange(num_clients), labels=[str(i) for i in range(num_clients)], fontsize=9)
    plt.legend(ncol=10, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Biểu đồ phân bố đã được lưu tại: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Tạo partition Non-IID Dirichlet cho MNIST trong Federated Learning")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hệ số tập trung Dirichlet alpha (mặc định: 0.5)")
    parser.add_argument("--num_clients", type=int, default=25, help="Số lượng client (mặc định: 25)")
    parser.add_argument("--seed", type=int, default=20261106, help="Hạt giống ngẫu nhiên (mặc định: 20261106)")
    parser.add_argument("--min_samples", type=int, default=10, help="Số lượng mẫu tối thiểu cho mỗi client (mặc định: 10)")
    parser.add_argument("--project_root", type=str, default=None, help="Đường dẫn thư mục dự án (tùy chọn)")
    parser.add_argument("--no_plot", action="store_true", help="Không vẽ biểu đồ phân bố")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè nếu file partition đã tồn tại")
    args = parser.parse_args()

    # 1. Xác định thư mục
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = get_default_project_root()

    data_dir = project_root / "data" / "mnist"
    partition_dir = data_dir / "partitions" / "Dirichlet"

    for folder in [data_dir, partition_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    # Đặt tên file theo chuẩn
    partition_filename = f"dirichlet_alpha{args.alpha}_{args.num_clients}_seed{args.seed}.pt"
    partition_path = partition_dir / partition_filename
    chart_filename = f"partition_dirichlet_alpha{args.alpha}_{args.num_clients}_clients_seed{args.seed}.png"
    chart_path = partition_dir / chart_filename

    # Kiểm tra file đã tồn tại
    if partition_path.exists() and not args.overwrite:
        print(f"File partition đã tồn tại: {partition_path}")
        print("Sử dụng cờ --overwrite nếu bạn muốn tạo lại.")
        return

    # 2. Tải MNIST train dataset
    print(f"Đang tải dữ liệu MNIST từ: {data_dir} ...")
    train_ds = datasets.MNIST(data_dir, train=True, download=True)
    train_labels = train_ds.targets.long()
    print(f"Đã tải {len(train_labels)} mẫu huấn luyện với 10 class.")

    # 3. Tạo partition Dirichlet
    print(f"Đang tạo partition Non-IID Dirichlet: alpha={args.alpha}, clients={args.num_clients}, seed={args.seed} ...")
    client_indices, client_class_counts = generate_dirichlet_partition(
        train_labels=train_labels,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        min_samples=args.min_samples,
    )

    # 4. Lưu partition vào thư mục partitions
    payload = {
        "client_indices": client_indices,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "alpha": args.alpha,
        "distribution": "dirichlet",
        "min_samples": args.min_samples,
        "class_counts": client_class_counts,
        "client_sizes": [len(x) for x in client_indices],
    }
    torch.save(payload, partition_path)
    print(f"Đã lưu partition thành công tại: {partition_path}")

    # 5. In thống kê
    client_sizes = [len(x) for x in client_indices]
    print("\n--- THỐNG KÊ PARTITION DIRICHLET ---")
    print(f"Số client: {args.num_clients}")
    print(f"Tham số alpha: {args.alpha}")
    print(f"Tổng số mẫu: {sum(client_sizes)}")
    print(f"Mẫu ít nhất của 1 client: {min(client_sizes)}")
    print(f"Mẫu nhiều nhất của 1 client: {max(client_sizes)}")
    print(f"Trung bình mỗi client: {np.mean(client_sizes):.1f} (std: {np.std(client_sizes):.1f})")
    print("Kích thước các client:", client_sizes)

    # 6. Vẽ biểu đồ nếu không bật cờ --no_plot
    if not args.no_plot:
        plot_and_save_distribution(
            client_class_counts=client_class_counts,
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=args.seed,
            chart_path=chart_path,
        )


if __name__ == "__main__":
    main()
