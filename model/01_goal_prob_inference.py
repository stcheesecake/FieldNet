import pickle
from collections import defaultdict
import numpy as np
import pandas as pd

PITCH_X_MAX = 105.0
MID_X = PITCH_X_MAX / 2.0

def type_name_from_type_result_str(s: str) -> str:
    return str(s).split("__", 1)[0].strip()

class RealtimeGoal10mPredictor:
    """
    프론트엔드/백엔드에서 이벤트 1개씩 들어올 때:
      - game_id별 상태(state) 유지
      - 같은 시각 t에 대해 양 팀 확률 2개 출력
    """
    def __init__(self, model_pkl_path: str):
        with open(model_pkl_path, "rb") as f:
            art = pickle.load(f)

        self.models = art["models"]
        self.feature_cols = art["feature_cols"]
        self.code_to_str = art["code_to_str"]
        self.code_to_type = art["code_to_type"]

        self.goal_codes = set(art["goal_codes"])
        self.own_goal_codes = set(art["own_goal_codes"])
        self.shot_codes = set(art["shot_codes"])

        self.target_sec = int(art["TARGET_TIME_MIN"] * 60)
        self.lookback_sec = int(art["LOOKBACK_MIN"] * 60)

        # game_id -> state
        self.state = {}

    @staticmethod
    def match_time_seconds(period_id, time_seconds):
        return int(time_seconds + max(int(period_id) - 1, 0) * 45 * 60)

    def _ensure_game(self, game_id, team_id):
        if game_id not in self.state:
            self.state[game_id] = {
                "teams": [],                         # 발견되는 순서대로 2팀
                "events": defaultdict(list),         # team -> list of (t, type_name, start_x_raw, dx_raw, period_id)
                "score": defaultdict(int),           # team -> goals so far
                "shot_stats": defaultdict(lambda: {"sum_endx":0.0, "n":0}),  # (team,period)-> running mean of shot end_x
            }
        st = self.state[game_id]
        if team_id not in st["teams"]:
            st["teams"].append(team_id)

    def _attack_right(self, st, team_id, period_id):
        key = (team_id, period_id)
        ss = st["shot_stats"][key]
        if ss["n"] == 0:
            return True  # 아직 모르면 일단 True로 가정
        return (ss["sum_endx"] / ss["n"]) > MID_X

    def _normalize_xy_online(self, st, team_id, period_id, start_x, end_x, dx):
        # shot 기반으로 공격방향 추정 후 flip
        if self._attack_right(st, team_id, period_id):
            return start_x, end_x, dx
        return (PITCH_X_MAX - start_x), (PITCH_X_MAX - end_x), (-dx)

    def _build_window_features(self, st, focal, opp, t_now):
        # type_name vocab은 학습 시 사용한 모든 type_name
        type_names = sorted(set(self.code_to_type.values()))

        def agg(team):
            cnt = {k: 0 for k in type_names}
            xs, dxs = [], []
            total = 0
            for (t, tn, x_raw, dx_raw, pid) in st["events"][team]:
                if t <= t_now - self.lookback_sec:
                    continue
                # 현재 시점 기준 공격방향으로 정규화해서 피처 일관성 유지
                x, _, dx = self._normalize_xy_online(st, team, pid, x_raw, x_raw, dx_raw)
                total += 1
                if tn in cnt:
                    cnt[tn] += 1
                xs.append(x)
                dxs.append(dx)

            feat = {f"cnt_{k}": v for k, v in cnt.items()}
            feat["total_events"] = total
            feat["avg_x"] = float(np.mean(xs)) if xs else 0.0
            feat["avg_dx"] = float(np.mean(dxs)) if dxs else 0.0
            return feat

        team_f = agg(focal)
        opp_f  = agg(opp)

        out = {}
        out.update({f"team_{k}": v for k, v in team_f.items()})
        out.update({f"opp_{k}": v for k, v in opp_f.items()})
        return out

    def update_and_predict(self, event: dict):
        """
        event dict 예시(프론트에서 넘어오는 형태):
        {
          "game_id": 126349,
          "period_id": 1,
          "time_seconds": 123.4,
          "team_id": 316,
          "start_x": 52.4, "end_x": 70.1,
          "dx": 17.7,
          "type_result": 40
        }
        반환:
          None (아직 팀 2개가 모이면)
          또는 {"game_id":..., "t":..., "probs": {teamA: pA, teamB: pB}}
        """
        gid = int(event["game_id"])
        team = int(event["team_id"])
        pid = int(event["period_id"])

        t_now = self.match_time_seconds(pid, float(event["time_seconds"]))
        tr = int(event["type_result"])

        self._ensure_game(gid, team)
        st = self.state[gid]

        # type_name
        tn = self.code_to_type.get(tr, "UNK")

        # shot이면 공격방향 추정용 러닝 평균 업데이트
        if tr in self.shot_codes:
            key = (team, pid)
            st["shot_stats"][key]["sum_endx"] += float(event["end_x"])
            st["shot_stats"][key]["n"] += 1

        # 득점 이벤트면 스코어 업데이트(own goal이면 상대 득점)
        if tr in self.goal_codes:
            credit_team = team
            if tr in self.own_goal_codes and len(st["teams"]) >= 2:
                other = st["teams"][0] if team == st["teams"][1] else st["teams"][1]
                credit_team = other
            st["score"][credit_team] += 1

        # 이벤트 저장(원본 좌표 저장 → 피처 생성 시점에 온라인 정규화)
        st["events"][team].append((
            t_now, tn,
            float(event["start_x"]),
            float(event["dx"]),
            pid
        ))

        # 오래된 이벤트 prune(팀별)
        for tid in st["teams"]:
            ev = st["events"][tid]
            # 앞에서부터 제거(단순 구현)
            while ev and ev[0][0] <= t_now - self.lookback_sec:
                ev.pop(0)

        # 팀 2개가 모여야 양 팀 확률 출력 가능
        if len(st["teams"]) < 2:
            return None

        A, B = st["teams"][0], st["teams"][1]

        probs = {}
        for focal, opp in [(A, B), (B, A)]:
            x = {
                "team_id": focal,
                "t": t_now,
                "minute": t_now / 60.0,
                "score_diff": int(st["score"][focal] - st["score"][opp]),
            }
            x.update(self._build_window_features(st, focal, opp, t_now))

            X = pd.DataFrame([x])[self.feature_cols]
            p = float(np.mean([m.predict(X)[0] for m in self.models]))
            probs[focal] = p

        return {"game_id": gid, "t": t_now, "probs": probs}
