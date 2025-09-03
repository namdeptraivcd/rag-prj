from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.config import Config

cfg = Config()


class HyPEEmbedder:
    def __init__(self):
        self.question_gen_prompt = """Analyze the input text and generate essential questions that, when answered, 
            capture the main points of the text. Each question should be one line, 
            without numbering or prefixes.\n\n
            Text:\n{chunk_text}\n\nQuestions:\n"""

        self.question_gen_prompt_template = PromptTemplate(
            input_variables=["chunk_text"],
            template=self.question_gen_prompt
        )
        
        self.question_chain = self.question_gen_prompt_template | cfg.llm | StrOutputParser()
    
    def generate_hypothetical_prompt_embeddings(self, chunk_text):
        # Generate questions for a chunk
        questions_text = self.question_chain.invoke({"chunk_text": chunk_text})
        
        # Debug 
        '''debug_index = 0
        import os
        file_name = os.path.basename(__file__)
        print(f"\n### Start debug {debug_index} in {file_name}")
        print("chunk_text: ", chunk_text)
        print(f"### End debug {debug_index} in {file_name}\n")'''
        
        # Remove newlines and split into individual questions
        questions = questions_text.replace("\n\n", "\n").split("\n")
        questions = [q.strip() for q in questions if q.strip()]
        
        # Debug 
        '''debug_index = 1
        import os
        file_name = os.path.basename(__file__)
        print(f"\n### Start debug {debug_index} in {file_name}")
        print("questions: ", questions)
        print(f"### End debug {debug_index} in {file_name}\n")'''
        
        # Embeed all questions above
        questions_embeddings = cfg.embeddings.embed_documents(questions)
        
        return questions_embeddings