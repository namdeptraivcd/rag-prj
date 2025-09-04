import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from pymilvus import MilvusClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
from src.config.config import Config

cfg = Config()


class GraphRAGProcessor:
    def __init__(self, dataset):
        self.milvus_client = MilvusClient(uri=cfg.milvus_uri, token=cfg.milvus_token)
        self.dataset = dataset
        
        # Collection names
        self.entity_collection_name = "entity_collection"
        self.relation_collection_name = "relation_collection"
        self.doc_collection_name = "doc_collection"
        
        # Data structure for graph relationships
        self.entities = []
        self.relations = []
        self.docs = []
        self.entityid_to_relationids = defaultdict(list)
        self.relationtd_to_docids = defaultdict(list)
        
        # Adjacency matrices for graph traversal
        self.entity_relation_adj = None
        self.entity_adj_target_degree = None
        self.relation_adj_target_degree = None
        self.entity_relation_adj_target_degree = None
        
        # Set up collections and data
        self.__set_up_collections_and_data()
    
    # SET UP COLLECTIONS AND DATA
    def __process_dataset(self, dataset: List[Dict]):
        for doc_id, doc in enumerate(dataset):
            doc_text = doc["doc"]
            triplets = doc["triplets"]
            self.docs.append(doc_text)
            
            for triplet in triplets:
                subject, predicate, obj = triplet
                
                # Add entities if not already present
                if subject not in self.entities:
                    self.entities.append(subject)
                if obj not in self.entities:
                    self.entities.append(obj)
                    
                # Create relation string and add to relations
                relation = f"{subject} {predicate} {obj}"
                if relation not in self.relations:
                    self.relations.append(relation)
                    
                    # Build entity-to-relation mappings
                    subject_id = self.entities.index(subject)
                    obj_id = self.entities.index(obj)
                    relation_id = len(self.relations) - 1
                    
                    self.entityid_to_relationids[subject_id].append(relation_id)
                    self.entityid_to_relationids[obj_id].append(relation_id)
                
                # Build relation-to-doc mappings
                relation_id = self.relations.index(relation)
                self.relationtd_to_docids[relation_id].append(doc_id)
    
    def __create_milvus_collection(self, collection_name):
        if self.milvus_client.has_collection(collection_name=collection_name):
            self.milvus_client.drop_collection(collection_name=collection_name)
            
        self.milvus_client.create_collection(
            collection_name=collection_name,
            dimension=cfg.embedding_dim,
            consistency_level="Strong"
        )
     
    def __insert_data_to_milvus(self, text_list: List[str], collection_name: str):
        batch_size = 512
        for row_id in tqdm(range(0, len(text_list), batch_size), desc=f"Inserting into {collection_name}"):
            batch_texts = text_list[row_id : row_id + batch_size]
            batch_embeddings = cfg.embeddings.embed_documents(batch_texts)
            
            batch_ids = [row_id + j for j in range(len(batch_texts))]
            batch_data = [
                {
                    "id": _id,
                    "text": text,
                    "vector": embedding 
                }
                for _id, text, embedding in zip(batch_ids, batch_texts, batch_embeddings)
            ]
            
            self.milvus_client.insert(collection_name=collection_name, data=batch_data)
    
    def __build_adjacency_matrices(self):
        self.entity_relation_adj = np.zeros((len(self.entities), len(self.relations)))
        for entity_id in range(len(self.entities)):
            for relation_id in self.entityid_to_relationids[entity_id]:
                self.entity_relation_adj[entity_id, relation_id] = 1
        
        # Convert to sparse matrix for efficiency
        self.entity_relation_adj = csr_matrix(self.entity_relation_adj)
        
        # Build 1-degree adjacency matrices
        entity_adj_1_degree = self.entity_relation_adj @ self.entity_relation_adj.T
        relation_adj_1_degree = self.entity_relation_adj.T @ self.entity_relation_adj
        
        # Compute target degree adjacency matrices
        self.entity_adj_target_degree = entity_adj_1_degree
        for _ in range(cfg.graph_rag_tartget_degree - 1):
            self.entity_adj_target_degree = self.entity_adj_target_degree @ entity_adj_1_degree.T
            
        self.relation_adj_target_degree = relation_adj_1_degree
        for _ in range(cfg.graph_rag_tartget_degree - 1):
            self.relation_adj_target_degree = self.relation_adj_target_degree @ relation_adj_1_degree.T 
            
        self.entity_relation_adj_target_degree = self.entity_adj_target_degree @ self.entity_relation_adj
            
    def __set_up_collections_and_data(self):
        # Process dataset
        self.__process_dataset(self.dataset)
        
        # Create Milvus collections
        self.__create_milvus_collection(self.entity_collection_name)
        self.__create_milvus_collection(self.relation_collection_name)
        self.__create_milvus_collection(self.doc_collection_name)
        
        # Insert data into collections
        self.__insert_data_to_milvus(self.entities, self.entity_collection_name)
        self.__insert_data_to_milvus(self.relations, self.relation_collection_name)
        self.__insert_data_to_milvus(self.docs, self.doc_collection_name)
        
        # Build adjacency matrices
        self.__build_adjacency_matrices()
    
    # QUERY GRAPH RAG
    def __retrieve_similar_entities_and_relations(self, query: str, query_entities: List[str], top_k: int = cfg.graph_rag_top_k_entites_or_relations):
        query_entity_embeddings = [cfg.embeddings.embed_query(entity) for entity in query_entities]
        
        # @TODO: find out how this 'search' method works
        retrieved_entities_list = self.milvus_client.search(
            collection_name=self.entity_collection_name,
            data=query_entity_embeddings,
            limit=top_k,
            output_fields=["id"]
        )
        
        # @TODO: find out why need [0] here
        # Relation-based retrieval using full query
        query_embedding = cfg.embeddings.embed_query(query)
        retrieved_relations = self.milvus_client.search(
            collection_name=self.relation_collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["id"],
        )[0]
        
        return retrieved_entities_list, retrieved_relations
    
    def __expand_subgraph(self, retrieved_entities_list, retrieved_relations) -> Tuple[Set[int], Set[int]]:
        expanded_relation_ids_from_entity = set()
        expanded_relation_ids_from_relation = set()
        
        # Expand from retrieved entities
        original_entity_ids = [
            retrieved_entity["entity"]["id"] 
            for retrieved_entities in retrieved_entities_list
            for retrieved_entity in retrieved_entities
        ]
        
        # @TODO: find out is [1] needed because we are using sparse matrix?
        for original_entity_id in original_entity_ids:
            expanded_relation_ids_from_entity.update(
                self.entity_relation_adj_target_degree[original_entity_id].nonzero()[1].tolist()
            )
        
        # Expand from retrieved relations
        original_relation_ids = [
            retrieved_relation["entity"]["id"] for retrieved_relation in retrieved_relations
        ]
        
        # @TODO: find out is [1] needed because we are using sparse matrix?
        for original_relation_id in original_relation_ids:
            expanded_relation_ids_from_relation.update(
                self.relation_adj_target_degree[original_relation_id].nonzero()[1].tolist()
            )
        
        return expanded_relation_ids_from_entity, expanded_relation_ids_from_relation
    
    def __rerank_relations(self, query: str, expanded_relations: List[str], expanded_relation_ids: List[int]) -> List[int]:
        expanded_relation_descriptions = []
        for _, (rel_id, rel) in enumerate(zip(expanded_relation_ids, expanded_relations)):
            expanded_relation_descriptions.append(f"[{rel_id}] {rel}")
        expanded_relation_descriptions_text = "\n".join(expanded_relation_descriptions).strip()
        
        # Debug 
        '''debug_index = 0
        import os
        file_name = os.path.basename(__file__)
        print(f"\n### Start debug {debug_index} in {file_name}")
        print(expanded_relation_descriptions_text)
        print(f"### End debug {debug_index} in {file_name}\n")'''
        
        # One-shot learning
        query_prompt_one_shot_input = """I will provide you with a list of relationship descriptions. Your task is to select 3 relationships that may be useful to answer the given question. Please return a JSON object containing your thought process and a list of the selected relationships in order of their relevance.
            Question:
            When was the mother of the leader of the Third Crusade born?

            Relationship descriptions:
            [1] Eleanor was born in 1122.
            [2] Eleanor married King Louis VII of France.
            [3] Eleanor was the Duchess of Aquitaine.
            [4] Eleanor participated in the Second Crusade.
            [5] Eleanor had eight children.
            [6] Eleanor was married to Henry II of England.
            [7] Eleanor was the mother of Richard the Lionheart.
            [8] Richard the Lionheart was the King of England.
            [9] Henry II was the father of Richard the Lionheart.
            [10] Henry II was the King of England.
            [11] Richard the Lionheart led the Third Crusade."""
        
        query_prompt_one_shot_output = """{"thought_process": "To answer the question about the birth of the mother of the leader of the Third Crusade, I first need to identify who led the Third Crusade and then determine who his mother was. After identifying his mother, I can look for the relationship that mentions her birth.", "useful_relationships": ["[11] Richard the Lionheart led the Third Crusade", "[7] Eleanor was the mother of Richard the Lionheart", "[1] Eleanor was born in 1122"]}"""
        
        query_prompt_template = """Question:
            {question}

            Relationship descriptions:
            {expanded_relation_descriptions_text}"""
        
        # Create prompt chain
        rerank_prompts = ChatPromptTemplate.from_messages([
            HumanMessage(query_prompt_one_shot_input),
            AIMessage(query_prompt_one_shot_output),
            HumanMessagePromptTemplate.from_template(query_prompt_template)
        ])
        
        rerank_chain = (rerank_prompts | cfg.llm.bind(response_format={"type": "json_object"})| JsonOutputParser())

        response = rerank_chain.invoke({
            "question": query,
            "expanded_relation_descriptions_text": expanded_relation_descriptions_text
        })
        
        # Extract relations' ids from LLM response
        rerank_relation_ids = []
        rerank_relation_descriptions = response["useful_relationships"] # -> List[str]
        for relation_description in rerank_relation_descriptions:
            start_idx = relation_description.find("[") + 1
            end_idx = relation_description.find("]")
            if start_idx > 0 and end_idx > start_idx:
                relation_id = int(relation_description[start_idx:end_idx])
                rerank_relation_ids.append(relation_id)
        
        return rerank_relation_ids
        
    def query_graph_rag(self, query: str, query_entities: List[str], final_top_k_docs=cfg.graph_rag_final_top_k_chunks) -> List[str]:
        # Retrieve similar entities and relations
        retrieved_entities_list, retrieved_relations = self.__retrieve_similar_entities_and_relations(query, query_entities)
        
        # Expand subgraph
        expanded_relation_ids_from_entity, expanded_relation_ids_from_relation = self.__expand_subgraph(retrieved_entities_list, retrieved_relations)
        
        # Merge expanded relations
        expanded_relation_ids = list(expanded_relation_ids_from_entity | expanded_relation_ids_from_relation)
        expanded_relations = [self.relations[id] for id in expanded_relation_ids]
        
        # LLM reranking
        rerank_relation_ids = self.__rerank_relations(query, expanded_relations, expanded_relation_ids)
        
        # Get final docs
        final_docs = []
        final_doc_ids = []
        for relation_id in rerank_relation_ids:
            for doc_id in self.relationtd_to_docids[relation_id]:
                if doc_id not in final_doc_ids:
                    final_doc_ids.append(doc_id)
                    final_docs.append(self.docs[doc_id])
        
        # Debug 
        '''debug_index = 1
        import os
        file_name = os.path.basename(__file__)
        print(f"\n### Start debug {debug_index} in {file_name}")
        print(final_docs[:final_top_k_docs])
        print(f"### End debug {debug_index} in {file_name}\n")'''
        
        return final_docs[:final_top_k_docs]
    
    # FOR EXPERIMENT PURPOSE
    def naive_rag_baseline(self):
        pass