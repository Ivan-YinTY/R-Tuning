from models import timer, timer_xl, moirai, moment, gpt4ts, ttm, time_llm, autotimes


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "timer": timer,           # Decoder-only     （公开，TimeR: decoder-only transformer）
            "timer_xl": timer_xl,     # Decoder-only √      （TimeR 的扩展版，仍是 decoder-only）
            "moirai": moirai,         # Encoder-only √      （官方定义，encoder-only attention）
            "moment": moment,         # Encoder-only      （MOMENT: purely encoder attention）
            "gpt4ts": gpt4ts,         # Decoder-only √      （使用 GPT2 backbone，是 decoder-only）
            "ttm": ttm,               # Encoder-only      （TAPE/TTM 系列是基于 time-aware encoder）
            "time_llm": time_llm,     # Decoder-only      （大模型 LLM 架构，一般为 decoder-only）
            "autotimes": autotimes,   # Encoder-only √      （AutoTimes 基于 N-BEATS / encoder stack）
                }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
