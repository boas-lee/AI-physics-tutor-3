# coding: utf-8

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
from datetime import datetime

# 환경 변수에서 OpenAI API 키를 불러옴
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    st.error("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
else:
    # 시스템 역할과 지침 설정
    system_role = """
    1. 모든 질문에 대한 답변은 한국어로 하고, 마크다운 형식으로 해야 합니다.
    2. 당신은 물리학자입니다.
    3. 당신은 30년 동안 대학생들에게 물리학을 가르치고 연구를 진행해 온 대학 교수입니다.
    4. 학생이 물리학과 관련된 질문을 하면, 예의 바르고 긍정적으로 답변합니다.
    5. 물리학과 관련된 질문에 성실히 답변하고, 물리학과 연관된 분야(과학, 예술, 사회현상, 철학 등)에도 답변합니다.
    6. 학생이 당신의 역할에 대해 질문하면, 당신이 물리학과 관련된 도움을 주기 위해 만들어진 AI임을 설명하며, 전문적인 톤을 유지합니다.
    7. 물리학과 관련이 없는 주제의 질문일 경우, 정중하게 답변을 거절하고 학생이 다른 질문을 하도록 안내합니다.
    8. 요청된 질문에 욕설, 비속어, 또는 사회적 규범에 위배되는 내용이 포함된 경우, 답변을 거부해야 합니다.
    """

    # 추가 지침
    instruction = """
    답변할 때는 아래의 지침을 엄격히 따르세요:
    1. 간단한 질문에 대한 답변은 5문장 이내로 하고, 70단어 이내로 유지하세요.
    2. 법칙이나 공식과 같은 상세한 설명이 필요한 경우에는 20문장 이내, 400단어 이내로 답변하세요.
    3. 수식이 포함된 답변은 LaTeX 형식을 사용하여 Markdown으로 제공하세요.
    4. 답변할 때 학생의 질문을 반복하지 말고, 오직 답변만 작성하세요.
    5. 학생에게 교수님이 말하는 것처럼 톤을 사용하세요.
       - 예시:
         - 학생: 물리학이란 무엇인가요?
         - 교수: 물리학은 자연의 기본 원리를 연구하는 학문이예요!
                 물체의 운동, 에너지, 힘 등을 다루죠. 
                 현대 과학과 기술의 기반입니다! 이해됐나요?
    6. 절대로 AI라는 사실을 밝히지 말고, 대학 교수로서 답변하세요.
    7. 질문에 욕설, 비속어, 사회적 규범에 어긋나는 행동이 포함된 경우 답변을 거부하세요.
    """

    # Streamlit 기본 설정
    st.set_page_config(page_title="🧲💥📡🌈 🤖 AI 물리 톡봇", layout="wide")
    st.title('🧲💥📡🌈 AI 물리 톡봇 👨‍🏫')
    st.header(' 물리에 대해서 물어보세요!')

    # ChatOpenAI 설정
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',  # 유효한 모델 이름 사용
        temperature=0.0, 
        max_tokens=4*1024,  # 최대 토큰 수 설정
        top_p=0.5,        # top_p 직접 설정
        n=1               # 한 번에 하나의 응답 생성
    )

    # 세션 상태에서 대화 기록 관리
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # 대화 생성 함수
    def conversational_chat(query):
        system_prompt = SystemMessage(content=system_role + instruction)

        # 현재까지의 대화 기록을 포함한 메시지 구성
        messages = [system_prompt] + st.session_state['history'] + [HumanMessage(content=query)]

        # LLM에 메시지 전달하여 응답 받기
        result = llm(messages=messages)

        # AIMessage로 응답 저장
        response = AIMessage(content=result.content)

        # 대화 기록을 업데이트
        st.session_state['history'].append(HumanMessage(content=query))  # 먼저 질문 추가
        st.session_state['history'].append(response)  # 그 뒤에 AI 응답 추가

        return response.content

    # 대화 기록 표시
    response_container = st.container()
    with response_container:
        for i in range(len(st.session_state['history'])):
            message_obj = st.session_state['history'][i]
            time_now = datetime.now().strftime("%H:%M")  # 현재 시간 기록
            if isinstance(message_obj, HumanMessage):
                # 사용자 입력 내용 배경색 변경 (파란색) + 시간 추가 + 오른쪽 정렬
                st.markdown(
                    f"""
                    <div style="background-color: #D4F1F4; padding: 10px; border-radius: 5px; text-align: right;">
                        🧑‍🎓 학생 ({time_now}): {message_obj.content}
                    </div>
                    """, unsafe_allow_html=True)
            elif isinstance(message_obj, AIMessage):
                # AI 답변 내용 배경색 변경 (주황색) + 시간 추가 + 왼쪽 정렬
                st.markdown(
                    f"""
                    <div style="background-color: #FFE5B4; padding: 10px; border-radius: 5px; text-align: left;">
                        👨‍🏫 AI 튜터 ({time_now}): {message_obj.content}
                    </div>
                    """, unsafe_allow_html=True)

    # 대화 입력 및 전송 버튼을 form으로 처리하여 엔터 키로도 전송 가능
    with st.container():
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("질문을 입력하세요:", placeholder="무엇이든 물어보세요!")
            submit_button = st.form_submit_button(label='Send')
            if submit_button and user_input:
                output = conversational_chat(user_input)

                # 질문과 답변 표시
                time_now = datetime.now().strftime("%H:%M")
                # 학생 입력 내용을 오른쪽에 배치
                st.markdown(
                    f"""
                    <div style="background-color: #D4F1F4; padding: 10px; border-radius: 5px; text-align: right;">
                        🧑‍🎓 학생 ({time_now}): {user_input}
                    </div>
                    """, unsafe_allow_html=True)
                # AI 답변 내용을 왼쪽에 배치
                st.markdown(
                    f"""
                    <div style="background-color: #FFE5B4; padding: 10px; border-radius: 5px; text-align: left;">
                        👨‍🏫 AI 튜터 ({time_now}): {output}
                    </div>
                    """, unsafe_allow_html=True)






