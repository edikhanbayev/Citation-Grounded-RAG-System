from app.models import RagEngine


class HybridRAGSystem:

    def __init__(self):

        self.rag = RagEngine()

    def generate(
        self,
        question
    ):

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Experiment A uses standalone questions.
        #
        # Therefore:
        #   chat_history = None
        #
        # This means your existing query condensation
        # function should return the question directly.
        # ----------------------------------------------------

        allowed_clearances = [
            "public",
            "internal",
            "restricted"
        ]

        (
            answer,
            citations,
            retrieved_chunks
        ) = self.rag.search_and_generate(

            question,

            allowed_clearances,

            chat_history=None,

            mode="hybrid"
        )

        return {

            "answer": answer,

            "citations": citations,

            "retrieved_chunks":
                retrieved_chunks
        }