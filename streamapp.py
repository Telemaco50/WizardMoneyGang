# %% [markdown]
# ## Project: Question-Answering on Private Documents

# %%
import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(),override=True)
import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone
# %%
st.title("GURU")
st.image('files/wiz.png')








#LOGICA APARTIR DE AQUI----------------------------------------------------------------------------------
# %%
# def load_document(file):
#     from langchain.document_loaders import PyPDFLoader
#     print(f'Loading {file}....')
#     loader=PyPDFLoader(file)
#     data= loader.load()
#     return data

# # %%
# def chunck_data(data,chunk_size=256):
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
#     chunks=text_splitter.split_documents(data)
#     return chunks

# # %%
# data= load_document('files/Manual del Dungeon Master Completo 5E.pdf')
# chunks=chunck_data(data,1500)
# # print(len(chunks))

# # %% [markdown]
# # ## Embedding y subirlo a Pinecone
# # 

# # %%
# def insert_or_fetch_embeddings(index_name):
#     import pinecone
#     from langchain.vectorstores import Pinecone
#     from langchain.embeddings import OpenAIEmbeddings
#     embeddings= OpenAIEmbeddings()
#     pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
#     if index_name  in pinecone.list_indexes():
#          print(f'El indice{index_name} ya existe, cargando embeddings...', end='')
#          vector_store= Pinecone.from_existing_index(index_name, embeddings)
#          print('OK')
#     else:
#           print(f'Creando el indice {index_name}...', end='')
#           pinecone.create_index(index_name, dimension=1536, metric='cosine')
#           vector_store=Pinecone.from_documents(chunks,embeddings,index_name=index_name)
#           print('OK')

#     return vector_store        
          



# # %%
# #Borrar indices
# def delete_pinecone_index(index_name='all'):
#     import pinecone
#     pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
#     if index_name == 'all':
#         indexes= pinecone.list_indexes()
#         print('Se van a borrar todos los indices...')
#         for index in indexes:
#             pinecone.delete_index(index)
#             print('Done')
#         else:
#             print(f'Borrando indice  {index_name}...', end='')
#             pinecone.delete_index(index_name)
#             print('Ok')

# %%
# delete_pinecone_index()

# %%
index_name='document-dnd'
# vector_store=insert_or_fetch_embeddings(index_name)

# %% [markdown]
# ### Se vienen preguntas 
embeddings= OpenAIEmbeddings()
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
vector_store=Pinecone.from_existing_index(index_name,embeddings)
# %%
def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm= ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriver=vector_store.as_retriever(search_type='similarity',search_kwargs={'k': 3})

    chain=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriver)
    answer=chain.run(q)
    return answer

def ask_with_memory(vector_store,q,chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    llm= ChatOpenAI( temperature=1)
    retriver=vector_store.as_retriever(search_type='similarity',search_kwargs={'k': 3})
    crc=ConversationalRetrievalChain.from_llm(llm,retriver)
    result=crc({'question':q,'chat_history':chat_history})
    chat_history.append((q,result['answer']))

    return result, chat_history

# %%
# q='¿Sobre que trata el documento?'
# answer= ask_and_get_answer(vector_store,q)
# print(answer)

# %%


# %%
import time
i= 1
st.write('Escribe Quit o Exit para parar la aplicacion')
while True:
    b= st.text_input(f'Pregunta #{i}: ')
    q=b
    i = i + 1
    if q.lower() in ['quit', 'exit']:
        print('Chao Pescao')
        time.sleep(2)
        break
    answer = ask_and_get_answer(vector_store,q) 
    st.write(f'\nRespuesta:{answer}')
    st.write(f'\n{"-" * 50}\n')

# %% [markdown]
# ### Running Code

# %%
# def print_embedding_cost(text):
#     import tiktoken
#     enc= tiktoken.encoding_for_model('text-embedding-ada-002')
#     total_tokens= sum([len(enc.encode(page.page_content))for page in text])
#     print(f'Numero total de Tokens{total_tokens}')
#     print(f'Mostrando coste en USD: {total_tokens / 1000 * 0.0004:.6f}')

# print_embedding_cost(chunks)


