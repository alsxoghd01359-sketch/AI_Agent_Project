"""Console TRPG DM chatbot.

The model (gpt-4o-mini) plays the Dungeon Master. It doesn't get the
rulebook or the character sheet stuffed into every prompt — instead it
decides for itself when to call `search_rulebook` (rules/spells/monsters)
or read/update `character_sheet` (HP, inventory, conditions...) via tool
calls, so context stays small and the numbers stay authoritative instead
of hallucinated.
"""
import json
import os
import re
import uuid

from dotenv import load_dotenv
from openai import OpenAI

from config import SCENARIO_PATH
from retrieval import search_rulebook
from character_sheets import (
    new_character_sheet, save_character, get_character,
    update_character, list_characters,
)

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4o-mini"
KEEP_RECENT_TURNS = 6  # raw turns kept verbatim; anything older gets folded into story_summary

SUMMARIZER_SYSTEM_PROMPT = """당신은 TRPG 세션 기록 요약가입니다. 기존 요약에 이번 대화 턴의 내용 중
이후 이야기 진행에 필요한 사실(사건, 장소, NPC, 결정, 습득한 정보 등)만 간결하게 반영해서 갱신된
요약을 출력하세요. 전투의 굴림 하나하나 같은 세부사항은 필요 없고, 결과와 줄거리에 남는 사실 위주로
쓰세요. 불필요하게 늘리지 말고, 기존 요약 문장도 필요 없어진 부분은 정리하면서 전체를 다시 쓰세요.
다른 설명 없이 갱신된 요약 텍스트만 출력하세요.
"""

SYSTEM_PROMPT = """당신은 던전즈 & 드래곤즈 5판(기초 규칙) TRPG의 던전 마스터(DM)입니다.
플레이어는 한 명이며, 캐릭터 ID는 "{character_id}" 입니다.

이번 게임의 시나리오 (플레이어가 직접 작성함):
{scenario}

규칙:
- 위 시나리오를 게임의 기반으로 삼아 진행하세요. 시나리오에 없는 세부사항은 자유롭게 채우되,
  시나리오에서 정한 배경, 분위기, 목표를 벗어나지 마세요. 시나리오와 무관하게 매번 비슷한 클리셰
  (예: 무조건 어둡고 축축한 동굴, 음산한 폐허)로 시작하거나 흘러가지 말고, 시나리오의 톤에 맞게
  다양하게 상황을 구성하세요.
- 시나리오에 "Act 1/2/3" 같은 명확한 사건 흐름이 적혀 있어도, 이는 강제로 완수해야 할 각본이
  아니라 참고용 기본 경로입니다. 플레이어가 그 흐름을 벗어나는 선택(예: 전투에서 도망치기, 사건을
  무시하고 다른 곳으로 가기, 예정된 만남을 피하기)을 하면, 우연한 사고나 갑자기 막힌 길, NPC의
  억지 설득 같은 부자연스러운 방법으로 원래 흐름으로 되돌리려 하지 마세요. 플레이어의 선택을
  그대로 존중하고, 그에 맞는 현실적인 결과(놓친 보상, NPC의 반응, 새로운 위험이나 기회 등)를 만들어
  이야기를 이어가세요. 플레이어의 자유의지가 항상 사전에 짜인 흐름보다 우선합니다.
- 규칙, 주문, 괴물 스탯 등 룰북 내용이 필요하면 반드시 search_rulebook 도구로 확인 후 답하세요. 추측으로 규칙을 지어내지 마세요.
- 전투나 사건으로 HP, 소지품, 상태이상 등이 바뀌면 반드시 update_character_sheet 도구로 실제 시트를 갱신하세요. 갱신하지 않으면 다음 판정에서 틀린 값을 쓰게 됩니다.
- 전투가 끝났다는 이유만으로 HP를 회복시키지 마세요. HP는 짧은 휴식/긴 휴식, 회복 주문, 물약 등
  명시적인 회복 수단을 사용했을 때만 늘어나야 하며, 그 외에는 전투 종료 후에도 깎인 상태 그대로
  유지되어야 합니다.
- 캐릭터의 현재 상태(HP, 능력치 등)가 필요하면 get_character_sheet 도구로 확인하세요.
- 판정이 필요한 상황(공격, 능력 판정, 내성 굴림 등)에서는 어떤 주사위를 굴려야 하는지, DC가 얼마인지 플레이어에게 명확히 알려주세요.
- 전투가 시작되면 update_combat_status 도구로 참가자 전원(플레이어, 동료 NPC, 적 몬스터)을 등록하세요.
  같은 종류의 몬스터가 여럿이면 "고블린A", "고블린B"처럼 구분되는 이름을 붙이세요. 누군가 공격을
  받거나, 회복하거나, 새로 등장하거나, 쓰러지거나, 전투가 끝날 때마다 즉시 update_combat_status를
  다시 호출해 전체 참가자 목록을 최신 상태로 갱신하세요 (부분 업데이트가 아니라 매번 전체를 다시
  보내세요). 공격 서술과 상태 갱신이 같은 턴에 함께 반영되어야 합니다. 플레이어 본인의 HP는
  update_character_sheet로 갱신한 값과 항상 일치시키세요.
- 생생하고 몰입감 있는 한국어로 서술하되, 장황하지 않게 핵심 위주로 진행하세요.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_rulebook",
            "description": "D&D 5e 기초 룰북에서 규칙/주문/괴물 정보를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색할 질문 또는 키워드 (예: '화염구 주문 효과', '기습 규칙')"},
                    "top_k": {"type": "integer", "description": "가져올 결과 수 (기본 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_character_sheet",
            "description": "캐릭터의 현재 시트(능력치, HP, 소지품 등)를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {"character_id": {"type": "string"}},
                "required": ["character_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_character_sheet",
            "description": "캐릭터 시트의 일부 필드를 갱신합니다 (예: HP 변화, 소지품 추가/제거, 상태이상 부여). "
                           "patch는 시트와 동일한 구조의 부분 객체이며, 기존 값 위에 덮어씌워집니다 "
                           "(예: {\"hit_points\": {\"max\": 8, \"current\": 5, \"temp\": 0}}).",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {"type": "string"},
                    "patch": {"type": "object", "description": "덮어쓸 필드들을 담은 객체"},
                },
                "required": ["character_id", "patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_combat_status",
            "description": "전투 참가자들의 체력/레벨 현황판을 갱신합니다. 전투 시작, HP 변화, 참가자 "
                           "등장/이탈/사망, 전투 종료 등 상태가 바뀔 때마다 호출하세요. 매번 부분 수정이 "
                           "아니라 그 시점의 전체 참가자 목록을 다시 보내주세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "in_combat": {"type": "boolean", "description": "현재 전투 중이면 true, 전투가 끝났으면 false"},
                    "combatants": {
                        "type": "array",
                        "description": "in_combat이 false면 빈 배열이어도 됩니다.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "표시용 이름. 동종 다수면 '고블린A', '고블린B'처럼 구분",
                                },
                                "side": {"type": "string", "enum": ["player", "ally", "enemy"]},
                                "hp_current": {"type": "integer"},
                                "hp_max": {"type": "integer"},
                                "level_or_cr": {
                                    "type": "string",
                                    "description": "플레이어/동료는 레벨(예: 'Lv.1'), 몬스터는 도전지수(예: 'CR 1/4')",
                                },
                            },
                            "required": ["name", "side", "hp_current", "hp_max"],
                        },
                    },
                },
                "required": ["in_combat", "combatants"],
            },
        },
    },
]


VERIFIER_SYSTEM_PROMPT = """당신은 TRPG DM 답변을 출력 직전에 검수하는 검수자입니다.
플레이어의 마지막 메시지와 DM 답변 초안을 보고 판단하세요.

- 먼저 플레이어의 메시지가 어떤 종류인지 파악하세요: (a) 행동/이동/대화 등 상황 진행 요청인지,
  (b) 규칙·주문·단어 뜻 설명 요청인지, (c) 피해량 등 계산 요청인지.
- (b), (c)처럼 게임 용어, 숫자, 계산식, 굴림 표기(예: 2d6+3) 등이 필요한 경우에는 그런 내용이
  나오는 것이 정상이니 그대로 두세요.
- (a)처럼 상황을 서술하는 답변인데, 판타지 세계관과 무관한 현실 단어(예: 원주율, 인터넷, 택배 등)나
  문맥에 맞지 않는 단어, 깨지거나 알아볼 수 없는 문자가 섞여 있다면 그 부분만 자연스러운 한국어로
  고쳐서 전체 답변을 다시 쓰세요. 나머지 내용과 어조는 최대한 그대로 유지하세요.
- ft, lbs, gp/sp/cp/ep/pp, DC, hp, AC, d4/d6/d8/d10/d12/d20 같은 룰북 표준 단위·약자 표기는
  영어라는 이유만으로 절대 한글로 바꾸지 마세요. 이런 표기를 "20ft"→"20피트"처럼 바꾸는 것도 잘못된
  수정입니다.
- 그 외에도 절대 손대지 마세요. 표현을 다듬거나, 문장을 매끄럽게 다시 쓰는 등의 "개선"은 하지
  마세요. 진짜 오류(문맥에 안 맞는 단어, 깨지거나 알아볼 수 없는 문자)가 없다면 입력받은 답변을
  토씨 하나 바꾸지 말고 그대로 출력하세요.
- 다른 설명 없이, 최종 답변 텍스트만 출력하세요.

예시:
입력: "당신은 조용한 원주율에 도착합니다." -> 출력: "당신은 조용한 마을에 도착합니다." (오류 수정)
입력: "20ft 반경에 8d6의 피해를 입습니다." -> 출력: "20ft 반경에 8d6의 피해를 입습니다." (그대로, 수정 금지)
"""


def verify_reply(user_message: str, draft_reply: str) -> str:
    if not draft_reply.strip():
        return draft_reply
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"플레이어 메시지:\n{user_message}\n\nDM 답변 초안:\n{draft_reply}"},
        ],
    )
    fixed = resp.choices[0].message.content
    return fixed.strip() if fixed else draft_reply


_combat_state = {"in_combat": False, "combatants": []}


def render_combat_status(character_id: str) -> str:
    if not _combat_state["in_combat"]:
        return ""

    # The model updates combat_status itself, but its player-side HP entry can lag
    # one turn behind the update_character_sheet call for the same event (two separate
    # tool calls it has to keep in sync by hand). The character sheet is the source of
    # truth, so force the player's row to match it instead of trusting the echoed number.
    sheet = get_character(character_id)
    if sheet:
        hp = sheet.get("hit_points") or {}
        for c in _combat_state["combatants"]:
            if c.get("side") == "player":
                c["name"] = sheet.get("name", c.get("name"))
                c["hp_current"] = hp.get("current", c.get("hp_current"))
                c["hp_max"] = hp.get("max", c.get("hp_max"))
                c["level_or_cr"] = f"Lv.{sheet.get('level')}"

    side_labels = [("player", "플레이어"), ("ally", "동료"), ("enemy", "적")]
    lines = ["--- 전투 상태 ---"]
    for side, label in side_labels:
        group = [c for c in _combat_state["combatants"] if c.get("side") == side]
        if not group:
            continue
        lines.append(f"[{label}]")
        for c in group:
            lvl = c.get("level_or_cr") or ""
            lvl_part = f" ({lvl})" if lvl else ""
            lines.append(f"  {c['name']}{lvl_part}: HP {c['hp_current']}/{c['hp_max']}")

    # Don't rely solely on the model remembering to call update_combat_status(in_combat=false)
    # when the fight is over — if every enemy is at 0 HP, end combat automatically so the
    # board doesn't linger on later, unrelated turns. Still show it this one last time so the
    # player sees the final blow land.
    enemies = [c for c in _combat_state["combatants"] if c.get("side") == "enemy"]
    if enemies and all(c.get("hp_current", 0) <= 0 for c in enemies):
        _combat_state["in_combat"] = False

    return "\n".join(lines)


def _run_tool(name: str, args: dict) -> str:
    try:
        if name == "search_rulebook":
            results = search_rulebook(args["query"], args.get("top_k", 5))
            trimmed = [
                {"name": r["name"], "chapter": r["chapter"], "section": r["section"], "text": r["text"]}
                for r in results
            ]
            return json.dumps(trimmed, ensure_ascii=False)

        if name == "get_character_sheet":
            sheet = get_character(args["character_id"])
            return json.dumps(sheet if sheet else {"error": "찾을 수 없음"}, ensure_ascii=False)

        if name == "update_character_sheet":
            updated = update_character(args["character_id"], args["patch"])
            return json.dumps(updated, ensure_ascii=False)

        if name == "update_combat_status":
            _combat_state["in_combat"] = bool(args.get("in_combat"))
            _combat_state["combatants"] = args.get("combatants", [])
            return json.dumps({"status": "ok", **_combat_state}, ensure_ascii=False)

        return json.dumps({"error": f"알 수 없는 도구: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _last_user_message(messages: list) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def get_dm_response(messages: list) -> str:
    while True:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.7
        )
        msg = resp.choices[0].message

        assistant_msg = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            draft = msg.content or ""
            final = verify_reply(_last_user_message(messages), draft)
            messages[-1]["content"] = final  # keep history consistent with what the player saw
            return final

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = _run_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


def _turn_to_narrative_text(turn_messages: list) -> str:
    lines = []
    for m in turn_messages:
        if m["role"] == "user":
            lines.append(f"플레이어: {m['content']}")
        elif m["role"] == "assistant" and m.get("content"):
            lines.append(f"DM: {m['content']}")
    return "\n".join(lines)


def summarize_turn(turn_messages: list, existing_summary: str) -> str:
    turn_text = _turn_to_narrative_text(turn_messages)
    if not turn_text.strip():
        return existing_summary
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"기존 요약:\n{existing_summary or '(없음)'}\n\n이번 대화 턴:\n{turn_text}\n\n갱신된 요약:",
            },
        ],
    )
    updated = resp.choices[0].message.content
    return updated.strip() if updated else existing_summary


def build_messages(character_id: str, scenario: str, story_summary: str, turn_history: list) -> list:
    system_content = SYSTEM_PROMPT.format(character_id=character_id, scenario=scenario)
    if story_summary:
        system_content += f"\n\n지금까지의 이야기 요약:\n{story_summary}"
    messages = [{"role": "system", "content": system_content}]
    for turn in turn_history:
        messages.extend(turn)
    return messages


def setup_scenario() -> str:
    if SCENARIO_PATH.exists():
        scenario = SCENARIO_PATH.read_text(encoding="utf-8").strip()
        if scenario:
            print(f"\n'{SCENARIO_PATH.name}'에서 시나리오를 불러왔습니다.")
            return scenario

    print(f"\n'{SCENARIO_PATH}' 파일이 없거나 비어 있습니다.")
    return "(시나리오 파일이 없습니다. 진부한 클리셰에 의존하지 말고 자유롭게 흥미로운 상황을 시작하세요.)"


def extract_opening_scene(scenario: str) -> str | None:
    """Pull out a hand-written "[오프닝 장면]" section so it's shown verbatim instead
    of being paraphrased by the model."""
    match = re.search(r"\[\s*오프닝\s*장면\s*\]\s*\n?(.*)", scenario, re.DOTALL)
    if not match:
        return None
    opening = match.group(1).strip()
    return opening or None


def setup_character() -> str:
    existing = list_characters()
    if existing:
        print("\n기존 캐릭터:")
        for c in existing:
            hp = c.get("hit_points", {})
            gender = c.get("gender") or "성별 미지정"
            print(f"  - {c['id']}: {c['name']} ({gender}, {c['race']} {c['class']} Lv.{c['level']}, "
                  f"HP {hp.get('current')}/{hp.get('max')})")
        choice = input("\n불러올 캐릭터 ID를 입력하거나, 새로 만들려면 Enter를 누르세요: ").strip()
        if choice:
            sheet = get_character(choice)
            if sheet:
                print(f"'{sheet['name']}' 캐릭터를 불러왔습니다.")
                return choice
            print("해당 ID를 찾을 수 없어 새로 만듭니다.")

    print("\n=== 새 캐릭터 생성 ===")
    name = input("이름: ").strip() or "이름 없는 모험자"
    gender = input("성별 (예: 남성, 여성, 없음 등): ").strip()
    race = input("종족 (예: 인간, 엘프, 드워프, 하플링): ").strip() or "인간"
    char_class = input("클래스 (예: 파이터, 로그, 위저드, 클레릭): ").strip() or "파이터"
    level_raw = input("레벨 (기본 1): ").strip()
    level = int(level_raw) if level_raw.isdigit() else 1
    hp_raw = input("최대 HP (모르면 Enter, DM이 나중에 정해줍니다): ").strip()
    max_hp = int(hp_raw) if hp_raw.isdigit() else 10

    sheet = new_character_sheet(name, race, char_class, level=level)
    sheet["gender"] = gender
    sheet["hit_points"] = {"max": max_hp, "current": max_hp, "temp": 0}

    character_id = f"char_{re.sub(r'[^0-9a-zA-Z가-힣]+', '_', name)}_{uuid.uuid4().hex[:6]}"
    save_character(character_id, sheet)
    print(f"'{name}' 캐릭터를 생성했습니다. (ID: {character_id})")
    return character_id


def main():
    print("=== TRPG DM 챗봇 (D&D 5e 기초 규칙) ===")
    character_id = setup_character()
    scenario = setup_scenario()

    story_summary = ""
    turn_history = []  # list of completed turns; each turn is a list of message dicts
    print("\n게임을 시작합니다. (종료: /quit)\n")

    opening = extract_opening_scene(scenario)
    if opening:
        print(f"DM> {opening}\n")
        turn_history.append([{"role": "assistant", "content": opening}])
    else:
        messages = build_messages(character_id, scenario, story_summary, turn_history)
        messages.append({"role": "user", "content": "게임을 시작해줘."})
        reply = get_dm_response(messages)
        print(f"DM> {reply}\n")
        status = render_combat_status(character_id)
        if status:
            print(status + "\n")
        turn_history.append(messages[1:])

    while True:
        user_input = input("플레이어> ").strip()
        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            print("게임을 종료합니다.")
            break

        messages = build_messages(character_id, scenario, story_summary, turn_history)
        messages.append({"role": "user", "content": user_input})

        try:
            reply = get_dm_response(messages)
        except Exception as e:
            print(f"[오류] {e}")
            continue

        print(f"\nDM> {reply}\n")
        status = render_combat_status(character_id)
        if status:
            print(status + "\n")

        base_len = 1 + sum(len(t) for t in turn_history)  # 1 == system message
        turn_history.append(messages[base_len:])

        if len(turn_history) > KEEP_RECENT_TURNS:
            oldest = turn_history.pop(0)
            story_summary = summarize_turn(oldest, story_summary)


if __name__ == "__main__":
    main()
