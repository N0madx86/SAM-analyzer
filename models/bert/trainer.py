import torch
import os

from torch.optim import AdamW

from transformers import (
    get_linear_schedule_with_warmup
)


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    epochs,
    learning_rate,
    weight_decay,
    warmup_ratio,
    patience,
    max_grad_norm,
    model_path
):

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    total_steps = len(train_loader) * epochs

    warmup_steps = int(
        total_steps * warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # AMP
    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_validation_loss = float("inf")

    patience_counter = 0

    for epoch in range(epochs):

        # ==================================================
        # Training
        # ==================================================

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True
            )

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True
            )

            labels = batch["labels"].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm
            )

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()

            running_loss += loss.item()

        train_loss = (
            running_loss /
            len(train_loader)
        )

        # ==================================================
        # Validation
        # ==================================================

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for batch in validation_loader:

                input_ids = batch["input_ids"].to(
                    device,
                    non_blocking=True
                )

                attention_mask = batch["attention_mask"].to(
                    device,
                    non_blocking=True
                )

                labels = batch["labels"].to(
                    device,
                    non_blocking=True
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp
                ):

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                validation_loss += outputs.loss.item()

        validation_loss /= len(validation_loader)

        # ==================================================
        # Best Model
        # ==================================================

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            patience_counter = 0

            os.makedirs(
                os.path.dirname(model_path),
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            status = " | Best model saved"

        else:

            patience_counter += 1

            status = (
                f" | Patience "
                f"{patience_counter}/{patience}"
            )

        print(
            f"Epoch {epoch + 1:02}/{epochs}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Validation Loss: {validation_loss:.4f}"
            f"{status}"
        )

        if patience_counter >= patience:

            print(
                "\nEarly stopping triggered."
            )

            break

    print("\nTraining complete.")