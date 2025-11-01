from rfdetr import RFDETRNano

if __name__ == "__main__":
    model = RFDETRNano(pretrain_weights=r"checkpoint_best_total.pth")

    model.train(
        dataset_dir="dataset",
        device="cpu",
        amp=False,
        epochs=12,          # hoặc ít hơn để test
        batch_size=4,       # nhỏ lại để nhanh hơn
        grad_accum_steps=4,  # chia hết cho batch_size (sub-batch = 1)
        num_workers=0,      # Windows + CPU nên để 0
        multi_scale=False,  # tắt cho nhanh
        output_dir="train_4",
        early_stopping=True,
    )
