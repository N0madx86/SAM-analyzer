import torch


def train_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    patience,
    model_path,
    device
):

    best_validation_loss = float("inf")
    counter = 0

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(x_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss /
            len(train_loader)
        )

        # Validation
        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for x_batch, y_batch in validation_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(x_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                validation_loss += loss.item()

        validation_loss /= len(validation_loader)

        model.train()

        # Scheduler
        scheduler.step(validation_loss)

        # Best checkpoint
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            counter = 0

            torch.save(
                model.state_dict(),
                model_path
            )

            status = " | Best model saved"

        else:

            counter += 1
            status = (
                f" | Patience "
                f"{counter}/{patience}"
            )

        print(
            f"Epoch {epoch + 1:02}/{epochs}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Validation Loss: {validation_loss:.4f}"
            f"{status}"
        )

        if counter >= patience:

            print("\nEarly stopping triggered.")
            break