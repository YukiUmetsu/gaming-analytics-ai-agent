from phoenix.otel import register

tracer_provider = register(
    project_name="udaplay",
    auto_instrument=True,
)

tracer = tracer_provider.get_tracer("udaplay")

def configure_observability():
    return register(
        project_name="udaplay",
        auto_instrument=True,
    )