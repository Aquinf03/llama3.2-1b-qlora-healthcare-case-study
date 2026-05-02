from aquin import attach_qlora


def run_with_aquin(trainer, model, optimizer, api_key, dataset):
    session = attach_qlora(
        model=model,
        optimizer=optimizer,
        api_key=api_key,
        project="qlora-research",
        run_name="experiment-1"
    )

    trainer.train(dataset, session=session)

    session.stop()
