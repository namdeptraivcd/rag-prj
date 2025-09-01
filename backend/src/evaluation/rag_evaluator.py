import time 
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.config.config import Config
from src.evaluation.grader import RelevanceGrader, FaithfulnessGrader, AnswerQualityGrader
from src.evaluation.evaluation_result import EvaluationResult

cfg = Config()


class RAGEvaluator:
    def __init__(self):
        self.llm = cfg.llm
        self.__setup_graders()
    
    def __setup_graders(self):
        # Relevance grader
        relevance_system = """You are a grader assessing relevance of retrieved documents to a user question. 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
        
        relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", relevance_system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
        '''self.relevance_grader = relevance_prompt | self.llm.with_structured_output(RelevanceGrader)'''
        def relevance_grader(document, question):
            prompt_text = relevance_prompt.invoke({
                "document": document,
                "question": question
            })
            result = self.llm.with_structured_output(RelevanceGrader).invoke(prompt_text)
            return result.binary_score
        self.relevance_grader = relevance_grader
        
        # Faithfulness Grader
        faithfulness_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        faithfulness_prompt = ChatPromptTemplate.from_messages([
            ("system", faithfulness_system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])

        '''self.faithfulness_grader = faithfulness_prompt | self.llm.with_structured_output(FaithfulnessGrader)'''
        def faithfulness_grader(documents, generation):
            prompt_text = faithfulness_prompt.invoke({"documents": documents, "generation": generation})
            result = self.llm.with_structured_output(FaithfulnessGrader).invoke(prompt_text)
            return result.binary_score
        self.faithfulness_grader = faithfulness_grader

        # Answer Quality Grader
        quality_system = """You are an expert evaluator assessing the quality of answers to questions.
            Rate the answer on a scale of 1-5 based on:
            - Completeness (does it fully answer the question?)
            - Accuracy (is the information correct?)
            - Clarity (is it well-written and understandable?)
            - Conciseness (is it appropriately detailed without being verbose?)
            
            5 = Excellent, 4 = Good, 3 = Average, 2 = Poor, 1 = Very Poor"""
        quality_prompt = ChatPromptTemplate.from_messages([
            ("system", quality_system),
            ("human", "Question: {question}\n\nAnswer: {answer}"),
        ])

        '''self.quality_grader = quality_prompt | self.llm.with_structured_output(AnswerQualityGrader)'''
        def answer_quality_grader(question, answer):
            prompt_text = quality_prompt.invoke({"question": question, "answer": answer})
            result = self.llm.with_structured_output(AnswerQualityGrader).invoke(prompt_text)
            return result.score
        self.answer_quality_grader = answer_quality_grader
        
    def evaluate_single_query(self, model, question: str) -> EvaluationResult:
        # Mesure response time
        start_time = time.time()
        
        # Add question to model state
        model.state["messages"].append(HumanMessage(content = question))
        
        # Query or respond
        messages, answer_type = model.query_or_respond()
        model.state["messages"].extend(messages)
        
        # Generate answer 
        model.generate()
        answer = model.state["answer"]
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Extract retrieved documents
        retrieved_docs = []
        for message in messages:
            if message.type == "tool":
                retrieved_docs.append(message.content)
        
        # Evaluate relevance of retrieved documents
        relevance_scores = []
        for retrieved_doc in retrieved_docs:
            relevance_score = self.relevance_grader(question, retrieved_doc)
            
            # Debug 
            '''debug_index = 1
            import os
            file_name = os.path.basename(__file__)
            print(f"\n### Start debug {debug_index} in {file_name}")
            print(type(relevance_score))
            print(f"### End debug {debug_index} in {file_name}\n")'''

            relevance_scores.append(relevance_score)
        
        overall_relevance_score = "no"
        for relevance_score in relevance_scores:
            if relevance_score == "yes":
                overall_relevance_score = "yes"
                break
        
        # Evaluate faithfulness of answer
        retrieved_docs_text = "\n\n".join(retrieved_docs)
        faithfulness_score = self.faithfulness_grader(retrieved_docs_text, answer)
        
        # Debug 
        '''debug_index = 2
        import os
        file_name = os.path.basename(__file__)
        print(f"\n### Start debug {debug_index} in {file_name}")
        print(type(faithfulness_score))
        print(f"### End debug {debug_index} in {file_name}\n")'''
        
        # Evaluate quality of answer
        answer_quality_score = self.answer_quality_grader(question, answer)
        
        # Debug 
        '''debug_index = 3
        import os
        file_name = os.path.basename(__file__)
        print(f"\n### Start debug {debug_index} in {file_name}")
        print(type(answer_quality_score))
        print(f"### End debug {debug_index} in {file_name}\n")'''
                
        return EvaluationResult(
            question=question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            response_time=response_time,
            relevance_score=overall_relevance_score,
            faithfulness_score=faithfulness_score,
            answer_quality_score=answer_quality_score
        )
    
    def evaluate_multiple_queries(self, model, questions: List[str]) -> List[EvaluationResult]:
        results = []
        
        for question in questions:
            # Reset model state for each question
            from langchain_core.messages import SystemMessage
            model.state["messages"] = []
            model.state["messages"].insert(0, SystemMessage(content=model.system_prompt))
            
            result = self.evaluate_single_query(model, question)
            results.append(result)
            
        return results
    
    def compute_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        if not results:
            return {}
        
        total_questions = len(results)
        avg_response_time = sum(r.response_time for r in results) / total_questions
        avg_relevance_score = sum(1 for r in results if r.relevance_score == "yes") / total_questions
        avg_faithfulness_score = sum(1 for r in results if r.faithfulness_score == "yes") / total_questions
        avg_answer_quality_score = sum(int(r.answer_quality_score) for r in results) / total_questions
        
        return {
            "average_response_time": avg_response_time,
            "average_relevance_score": avg_relevance_score,
            "average_faithfulness_score": avg_faithfulness_score,
            "average_answer_quality_score": avg_answer_quality_score,
            "total_questions": total_questions
        }
    