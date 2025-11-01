from rfdetr import RFDETRNano

if __name__ == "__main__":
    model = RFDETRNano()

    model.train(
        dataset_dir="dataset",
        device="cpu",
        amp=False,
        epochs=10,
        batch_size=2,
        grad_accum_steps=8,
        lr=1e-4,
        output_dir="train"
    )
