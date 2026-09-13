from lib.memory import (
    LongTermMemory,
    MemoryFragment,
)


class FakeVectorStore:
    def __init__(self):
        self.documents = []

    def add(self, document):
        self.documents.append(document)

    def query(
        self,
        query_texts,
        n_results=3,
        where=None,
        **kwargs,
    ):
        matching_documents = []

        for document in self.documents:
            metadata = document.metadata

            owner = metadata.get("owner")
            namespace = metadata.get("namespace")

            expected_owner = None
            expected_namespace = None

            for condition in where.get("$and", []):
                if "owner" in condition:
                    expected_owner = condition["owner"]["$eq"]

                if "namespace" in condition:
                    expected_namespace = condition["namespace"]["$eq"]

            if (
                owner == expected_owner
                and namespace == expected_namespace
            ):
                matching_documents.append(document)

        matching_documents = matching_documents[:n_results]

        return {
            "documents": [
                [
                    document.content
                    for document in matching_documents
                ]
            ],
            "metadatas": [
                [
                    document.metadata
                    for document in matching_documents
                ]
            ],
            "distances": [
                [
                    0.1
                    for _ in matching_documents
                ]
            ],
        }


class FakeVectorStoreManager:
    def __init__(self):
        self.store = FakeVectorStore()
        self.requested_store = None

    def get_or_create_store(self, name):
        self.requested_store = name
        return self.store


def test_long_term_memory_uses_existing_persistent_store():
    manager = FakeVectorStoreManager()

    LongTermMemory(manager)

    assert manager.requested_store == "long_term_memory"


def test_long_term_memory_registers_fragment():
    manager = FakeVectorStoreManager()
    memory = LongTermMemory(manager)

    fragment = MemoryFragment(
        content="GTA VI is developed by Rockstar Games.",
        owner="udaplay",
        namespace="game_knowledge",
    )

    memory.register(
        fragment,
        metadata={
            "source_url": "https://example.com/gta6",
            "source_title": "GTA VI",
            "source_type": "web",
        },
    )

    assert len(manager.store.documents) == 1

    document = manager.store.documents[0]

    assert document.content == (
        "GTA VI is developed by Rockstar Games."
    )

    assert document.metadata["owner"] == "udaplay"
    assert document.metadata["namespace"] == "game_knowledge"
    assert document.metadata["source_url"] == "https://example.com/gta6"
    assert document.metadata["source_type"] == "web"


def test_long_term_memory_can_find_registered_memory():
    manager = FakeVectorStoreManager()
    memory = LongTermMemory(manager)

    memory.register(
        MemoryFragment(
            content="GTA VI is developed by Rockstar Games.",
            owner="udaplay",
            namespace="game_knowledge",
        )
    )

    result = memory.search(
        query_text="What is Rockstar developing?",
        owner="udaplay",
        namespace="game_knowledge",
    )

    assert len(result.fragments) == 1
    assert result.fragments[0].content == (
        "GTA VI is developed by Rockstar Games."
    )

    assert result.metadata["distances"] == [0.1]


def test_long_term_memory_does_not_return_different_owner():
    manager = FakeVectorStoreManager()
    memory = LongTermMemory(manager)

    memory.register(
        MemoryFragment(
            content="Information belonging to another owner.",
            owner="other-agent",
            namespace="game_knowledge",
        )
    )

    result = memory.search(
        query_text="information",
        owner="udaplay",
        namespace="game_knowledge",
    )

    assert result.fragments == []


def test_long_term_memory_separates_namespaces():
    manager = FakeVectorStoreManager()
    memory = LongTermMemory(manager)

    memory.register(
        MemoryFragment(
            content="Current GTA VI information.",
            owner="udaplay",
            namespace="game_knowledge",
        )
    )

    result = memory.search(
        query_text="GTA VI",
        owner="udaplay",
        namespace="user_preferences",
    )

    assert result.fragments == []