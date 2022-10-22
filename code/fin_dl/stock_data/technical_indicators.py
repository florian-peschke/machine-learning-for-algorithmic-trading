import typing as t

import attr
import attrs
import numpy as np
import pandas as pd
from talib import abstract, MA_Type


@attrs.define
class ArgumentsForAbstractAPI:
    frame: pd.DataFrame = attr.field(kw_only=True)
    columns: t.List[str] = attr.field(kw_only=True)
    parameters: t.Dict[str, t.Any] = attr.field(kw_only=True)

    def __get_values(self) -> t.Iterator[np.ndarray]:
        for name in self.columns:
            yield self.frame.loc[:, name].values

    @property
    def kwargs(self) -> t.Dict[str, t.Any]:
        return self.parameters

    @property
    def price(self) -> t.Tuple[np.ndarray, ...]:
        return tuple(list(self.__get_values()))


@attrs.define
class TechnicalIndicators:
    label_open: str = "open"
    label_high: str = "high"
    label_low: str = "low"
    label_close: str = "close"
    label_volume: str = "volume"

    timeperiod: int = 20
    timeperiod1: int = 5
    timeperiod2: int = 10
    timeperiod3: int = 20

    fastperiod: int = 3
    slowperiod: int = 10

    minperiod: int = 2
    maxperiod: int = 30

    nbdev: float = 1.0

    acceleration: float = 0.0
    maximum: float = 0.0

    startvalue: float = 0.0
    offsetonreverse: float = 0.0
    accelerationinitlong: float = 0.0
    accelerationlong: float = 0.0
    accelerationmaxlong: float = 0.0
    accelerationinitshort: float = 0.0
    accelerationshort: float = 0.0
    accelerationmaxshort: float = 0.0

    vfactor: float = 0.0

    penetration: float = 0.0

    matype: int = MA_Type.T3

    def calculate_all(self, frame: pd.DataFrame) -> pd.DataFrame:
        callables: t.List[t.Callable] = [
            self.overlap_studies,
            self.cycle_indicators,
            self.momentum_indicators,
            self.pattern_recognition,
            self.price_transform,
            self.statistic_functions,
            self.volatility_indicators,
            self.volume_indicators,
        ]
        return self.__recursive_calculation(frame=frame, callables=callables)

    def __recursive_calculation(self, frame: pd.DataFrame, callables: t.List[t.Callable]) -> pd.DataFrame:
        if len(callables) == 1:
            return callables[0](frame=frame)
        new_frame: pd.DataFrame = callables[0](frame=frame)
        return self.__recursive_calculation(frame=new_frame, callables=callables[1:])

    @staticmethod
    def __calculate_values(indicator_name: str, frame: pd.DataFrame, kwargs: t.Dict[str, t.Any]) -> np.ndarray:
        fun: t.Callable = abstract.Function(indicator_name.lower())
        arguments: ArgumentsForAbstractAPI = ArgumentsForAbstractAPI(
            frame=frame, columns=kwargs["columns"], parameters=kwargs["parameters"]
        )
        return fun(*arguments.price, **arguments.kwargs)

    def __yield_technical_indicators(
        self, frame: pd.DataFrame, function_names_with_arguments: t.Dict[str, t.Any]
    ) -> t.Iterator[pd.Series]:
        for indicator_name, arguments in function_names_with_arguments.items():
            yield pd.Series(
                self.__calculate_values(frame=frame, indicator_name=indicator_name, kwargs=arguments),
                name=indicator_name,
                index=frame.index,
            )

    def __insert_technical_indicators(
        self, frame: pd.DataFrame, function_names_with_arguments: t.Dict[str, t.Any]
    ) -> pd.DataFrame:
        technical_indicators: pd.DataFrame = pd.concat(
            list(
                self.__yield_technical_indicators(
                    frame=frame, function_names_with_arguments=function_names_with_arguments
                )
            ),
            axis=1,
            names=list(function_names_with_arguments.keys()),
        )
        return pd.concat([frame, technical_indicators], axis=1)

    def __get_indicators_by_type(
        self,
        function_names_with_arguments_callable: t.Callable,
        frame: pd.DataFrame,
        custom_arguments: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> pd.DataFrame:
        function_names_with_arguments: t.Dict[str, t.Dict] = function_names_with_arguments_callable()
        if custom_arguments is not None:
            function_names_with_arguments.update(custom_arguments)
        return self.__insert_technical_indicators(
            frame=frame, function_names_with_arguments=function_names_with_arguments
        )

    def overlap_studies(
        self, frame: pd.DataFrame, custom_arguments: t.Optional[t.Dict[str, t.Any]] = None
    ) -> pd.DataFrame:
        return self.__get_indicators_by_type(
            function_names_with_arguments_callable=self.__overlap_studies_arguments,
            frame=frame,
            custom_arguments=custom_arguments,
        )

    def cycle_indicators(
        self, frame: pd.DataFrame, custom_arguments: t.Optional[t.Dict[str, t.Any]] = None
    ) -> pd.DataFrame:
        return self.__get_indicators_by_type(
            function_names_with_arguments_callable=self.__cycle_indicators_arguments,
            frame=frame,
            custom_arguments=custom_arguments,
        )

    def momentum_indicators(
        self, frame: pd.DataFrame, custom_arguments: t.Optional[t.Dict[str, t.Any]] = None
    ) -> pd.DataFrame:
        return self.__get_indicators_by_type(
            function_names_with_arguments_callable=self.__momentum_indicators_arguments,
            frame=frame,
            custom_arguments=custom_arguments,
        )

    def pattern_recognition(
        self, frame: pd.DataFrame, custom_arguments: t.Optional[t.Dict[str, t.Any]] = None
    ) -> pd.DataFrame:
        return self.__get_indicators_by_type(
            function_names_with_arguments_callable=self.__pattern_recognition_arguments,
            frame=frame,
            custom_arguments=custom_arguments,
        )

    def price_transform(
        self, frame: pd.DataFrame, custom_arguments: t.Optional[t.Dict[str, t.Any]] = None
    ) -> pd.DataFrame:
        return self.__get_indicators_by_type(
            function_names_with_arguments_callable=self.__price_transform_arguments,
            frame=frame,
            custom_arguments=custom_arguments,
        )

    def statistic_functions(
        self, frame: pd.DataFrame, custom_arguments: t.Optional[t.Dict[str, t.Any]] = None
    ) -> pd.DataFrame:
        return self.__get_indicators_by_type(
            function_names_with_arguments_callable=self.__statistic_functions_arguments,
            frame=frame,
            custom_arguments=custom_arguments,
        )

    def volatility_indicators(
        self, frame: pd.DataFrame, custom_arguments: t.Optional[t.Dict[str, t.Any]] = None
    ) -> pd.DataFrame:
        return self.__get_indicators_by_type(
            function_names_with_arguments_callable=self.__volatility_indicators_arguments,
            frame=frame,
            custom_arguments=custom_arguments,
        )

    def volume_indicators(
        self, frame: pd.DataFrame, custom_arguments: t.Optional[t.Dict[str, t.Any]] = None
    ) -> pd.DataFrame:
        return self.__get_indicators_by_type(
            function_names_with_arguments_callable=self.__volume_indicators_arguments,
            frame=frame,
            custom_arguments=custom_arguments,
        )

    def __overlap_studies_arguments(self) -> t.Dict[str, t.Dict]:
        return {
            # Double Exponential Moving Average – http://www.tadoc.org/indicator/DEMA.htm
            "DEMA": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Exponential Moving Average – http://www.tadoc.org/indicator/EMA.htm
            "EMA": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Hilbert Transform - Instantaneous Trendline – http://www.tadoc.org/indicator/HT_TRENDLINE.htm
            "HT_TRENDLINE": dict(columns=[self.label_close], parameters={}),
            # Kaufman Adaptive Moving Average – http://www.tadoc.org/indicator/KAMA.htm
            "KAMA": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Moving average
            "MA": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod, matype=self.matype)),
            # MidPoint over period – http://www.tadoc.org/indicator/MIDPOINT.htm
            "MIDPOINT": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Midpoint Price over period – http://www.tadoc.org/indicator/MIDPRICE.htm
            "MIDPRICE": dict(columns=[self.label_high, self.label_low], parameters=dict(timeperiod=self.timeperiod)),
            # Parabolic SAR – http://www.tadoc.org/indicator/SAR.htm
            "SAR": dict(
                columns=[self.label_high, self.label_low],
                parameters=dict(acceleration=self.acceleration, maximum=self.maximum),
            ),
            # Parabolic SAR - Extended
            "SAREXT": dict(
                columns=[self.label_high, self.label_low],
                parameters=dict(
                    startvalue=self.startvalue,
                    offsetonreverse=self.offsetonreverse,
                    accelerationinitlong=self.accelerationinitlong,
                    accelerationlong=self.accelerationlong,
                    accelerationmaxlong=self.accelerationmaxlong,
                    accelerationinitshort=self.accelerationinitshort,
                    accelerationshort=self.accelerationshort,
                    accelerationmaxshort=self.accelerationmaxshort,
                ),
            ),
            # Simple Moving Average
            "SMA": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Triple Exponential Moving Average (T3) – http://www.tadoc.org/indicator/T3.htm
            "T3": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod, vfactor=self.vfactor)),
            # Triple Exponential Moving Average – http://www.tadoc.org/indicator/TEMA.htm
            "TEMA": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Triangular Moving Average – http://www.tadoc.org/indicator/TRIMA.htm
            "TRIMA": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Weighted Moving Average – http://www.tadoc.org/indicator/WMA.htm
            "WMA": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
        }

    def __cycle_indicators_arguments(self) -> t.Dict[str, t.Dict]:
        return {
            # Hilbert Transform - Dominant Cycle Period – http://www.tadoc.org/indicator/HT_DCPERIOD.htm
            "HT_DCPERIOD": dict(columns=[self.label_close], parameters={}),
            # Hilbert Transform - Dominant Cycle Phase – http://www.tadoc.org/indicator/HT_DCPHASE.htm
            "HT_DCPHASE": dict(columns=[self.label_close], parameters={}),
        }

    def __momentum_indicators_arguments(self) -> t.Dict[str, t.Dict]:
        return {
            # Average Directional Movement Index – http://www.tadoc.org/indicator/ADX.htm
            "ADX": dict(
                columns=[self.label_high, self.label_low, self.label_close], parameters=dict(timeperiod=self.timeperiod)
            ),
            # Average Directional Movement Index Rating – http://www.tadoc.org/indicator/ADXR.htm
            "ADXR": dict(
                columns=[self.label_high, self.label_low, self.label_close], parameters=dict(timeperiod=self.timeperiod)
            ),
            # Absolute Price Oscillator – http://www.tadoc.org/indicator/APO.htm
            "APO": dict(
                columns=[self.label_close],
                parameters=dict(fastperiod=self.fastperiod, slowperiod=self.slowperiod, matype=self.matype),
            ),
            # Aroon Oscillator – http://www.tadoc.org/indicator/AROONOSC.htm
            "AROONOSC": dict(columns=[self.label_high, self.label_low], parameters=dict(timeperiod=self.timeperiod)),
            # Balance Of Power – http://www.tadoc.org/indicator/BOP.htm
            "BOP": dict(columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}),
            # Commodity Channel Index – http://www.tadoc.org/indicator/CCI.htm
            "CCI": dict(
                columns=[self.label_high, self.label_low, self.label_close], parameters=dict(timeperiod=self.timeperiod)
            ),
            # Chande Momentum Oscillator – http://www.tadoc.org/indicator/CMO.htm
            "CMO": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Directional Movement Index – http://www.tadoc.org/indicator/DX.htm
            "DX": dict(
                columns=[self.label_high, self.label_low, self.label_close], parameters=dict(timeperiod=self.timeperiod)
            ),
            # Money Flow Index – http://www.tadoc.org/indicator/MFI.htm
            "MFI": dict(
                columns=[self.label_high, self.label_low, self.label_close, self.label_volume],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Minus Directional Indicator – http://www.tadoc.org/indicator/MINUS_DI.htm
            "MINUS_DI": dict(
                columns=[self.label_high, self.label_low, self.label_close],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Minus Directional Movement – http://www.tadoc.org/indicator/MINUS_DM.htm
            "MINUS_DM": dict(
                columns=[self.label_high, self.label_low],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Momentum – http://www.tadoc.org/indicator/MOM.htm
            "MOM": dict(
                columns=[self.label_close],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Plus Directional Indicator – http://www.tadoc.org/indicator/PLUS_DI.htm
            "PLUS_DI": dict(
                columns=[self.label_high, self.label_low, self.label_close],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Plus Directional Movement – http://www.tadoc.org/indicator/PLUS_DM.htm
            "PLUS_DM": dict(
                columns=[self.label_high, self.label_low],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Percentage Price Oscillator – http://www.tadoc.org/indicator/PPO.htm
            "PPO": dict(
                columns=[self.label_close],
                parameters=dict(fastperiod=self.fastperiod, slowperiod=self.slowperiod, matype=self.matype),
            ),
            # Rate of change – http://www.tadoc.org/indicator/ROC.htm
            "ROC": dict(
                columns=[self.label_close],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Rate of change Percentage – http://www.tadoc.org/indicator/ROCP.htm
            "ROCP": dict(
                columns=[self.label_close],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Rate of change ratio – http://www.tadoc.org/indicator/ROCR.htm
            "ROCR": dict(
                columns=[self.label_close],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Relative Strength Index – http://www.tadoc.org/indicator/RSI.htm
            "RSI": dict(
                columns=[self.label_close],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA – http://www.tadoc.org/indicator/TRIX.htm
            "TRIX": dict(
                columns=[self.label_close],
                parameters=dict(timeperiod=self.timeperiod),
            ),
            # Ultimate Oscillator – http://www.tadoc.org/indicator/ULTOSC.htm
            "ULTOSC": dict(
                columns=[self.label_high, self.label_low, self.label_close],
                parameters=dict(
                    timeperiod1=self.timeperiod1, timeperiod2=self.timeperiod2, timeperiod3=self.timeperiod3
                ),
            ),
            # Williams' %R – http://www.tadoc.org/indicator/WILLR.htm
            "WILLR": dict(
                columns=[self.label_high, self.label_low, self.label_close],
                parameters=dict(timeperiod=self.timeperiod),
            ),
        }

    def __pattern_recognition_arguments(self) -> t.Dict[str, t.Dict]:
        return {
            # Two Crows
            "CDL2CROWS": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Three Black Crows
            "CDL3BLACKCROWS": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Three Inside Up/Down
            "CDL3INSIDE": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Three-Line Strike
            "CDL3LINESTRIKE": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Three Outside Up/Down
            "CDL3OUTSIDE": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Three Stars In The South
            "CDL3STARSINSOUTH": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Three Advancing White Soldiers
            "CDL3WHITESOLDIERS": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Abandoned Baby
            "CDLABANDONEDBABY": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close],
                parameters=dict(penetration=self.penetration),
            ),
            # Advance Block
            "CDLADVANCEBLOCK": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Belt-hold
            "CDLBELTHOLD": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Breakaway
            "CDLBREAKAWAY": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Closing Marubozu
            "CDLCLOSINGMARUBOZU": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Concealing Baby Swallow
            "CDLCONCEALBABYSWALL": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Counterattack
            "CDLCOUNTERATTACK": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Dark Cloud Cover
            "CDLDARKCLOUDCOVER": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close],
                parameters=dict(penetration=self.penetration),
            ),
            # Doji
            "CDLDOJI": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Doji Star
            "CDLDOJISTAR": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Dragonfly Doji
            "CDLDRAGONFLYDOJI": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Engulfing Pattern
            "CDLENGULFING": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Evening Doji Star
            "CDLEVENINGDOJISTAR": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close],
                parameters=dict(penetration=self.penetration),
            ),
            # Evening Star
            "CDLEVENINGSTAR": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close],
                parameters=dict(penetration=self.penetration),
            ),
            # Up/Down-gap side-by-side white lines
            "CDLGAPSIDESIDEWHITE": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Gravestone Doji
            "CDLGRAVESTONEDOJI": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Hammer
            "CDLHAMMER": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Hanging Man
            "CDLHANGINGMAN": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Harami Pattern
            "CDLHARAMI": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Harami Cross Pattern
            "CDLCOUNTERATTACK": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Counterattack
            "CDLHARAMICROSS": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # High-Wave Candle
            "CDLHIGHWAVE": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Hikkake Pattern
            "CDLHIKKAKE": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Modified Hikkake Pattern
            "CDLHIKKAKEMOD": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Homing Pigeon
            "CDLHOMINGPIGEON": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Identical Three Crows
            "CDLIDENTICAL3CROWS": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # In-Neck Pattern
            "CDLINNECK": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Inverted Hammer
            "CDLINVERTEDHAMMER": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Kicking
            "CDLKICKING": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Kicking - bull/bear determined by the longer marubozu
            "CDLKICKINGBYLENGTH": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Ladder Bottom
            "CDLLADDERBOTTOM": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Long Legged Doji
            "CDLLONGLEGGEDDOJI": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Long Line Candle
            "CDLLONGLINE": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Marubozu
            "CDLMARUBOZU": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Matching Low
            "CDLMATCHINGLOW": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Mat Hold
            "CDLLONGLEGGEDDOJI": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close],
                parameters=dict(penetration=self.penetration),
            ),
            # Morning Doji Star
            "CDLMORNINGDOJISTAR": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close],
                parameters=dict(penetration=self.penetration),
            ),
            # Morning Star
            "CDLMORNINGSTAR": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close],
                parameters=dict(penetration=self.penetration),
            ),
            # On-Neck Pattern
            "CDLONNECK": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Piercing Pattern
            "CDLPIERCING": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Rickshaw Man
            "CDLRICKSHAWMAN": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Rising/Falling Three Methods
            "CDLRISEFALL3METHODS": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Separating Lines
            "CDLSEPARATINGLINES": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Shooting Star
            "CDLSHOOTINGSTAR": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Short Line Candle
            "CDLSHORTLINE": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Spinning Top
            "CDLSPINNINGTOP": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Stalled Pattern
            "CDLSTALLEDPATTERN": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Stick Sandwich
            "CDLSTICKSANDWICH": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Takuri (Dragonfly Doji with very long lower shadow)
            "CDLTAKURI": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Tasuki Gap
            "CDLTASUKIGAP": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Thrusting Pattern
            "CDLTHRUSTING": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Tristar Pattern
            "CDLTRISTAR": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Unique 3 River
            "CDLUNIQUE3RIVER": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Upside Gap Two Crows
            "CDLUPSIDEGAP2CROWS": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Upside/Downside Gap Three Methods
            "CDLXSIDEGAP3METHODS": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
        }

    def __price_transform_arguments(self) -> t.Dict[str, t.Dict]:
        return {
            # Average Price – http://www.tadoc.org/indicator/AVGPRICE.htm
            "AVGPRICE": dict(
                columns=[self.label_open, self.label_high, self.label_low, self.label_close], parameters={}
            ),
            # Median Price – http://www.tadoc.org/indicator/MEDPRICE.htm
            "MEDPRICE": dict(columns=[self.label_high, self.label_low], parameters={}),
            # Typical Price – http://www.tadoc.org/indicator/TYPPRICE.htm
            "TYPPRICE": dict(columns=[self.label_high, self.label_low, self.label_close], parameters={}),
            # Weighted Close Price – http://www.tadoc.org/indicator/WCLPRICE.htm
            "WCLPRICE": dict(columns=[self.label_high, self.label_low, self.label_close], parameters={}),
        }

    def __statistic_functions_arguments(self) -> t.Dict[str, t.Dict]:
        return {
            # Beta – http://www.tadoc.org/indicator/BETA.htm
            "BETA": dict(columns=[self.label_high, self.label_low], parameters=dict(timeperiod=self.timeperiod)),
            # Pearson's Correlation Coefficient (r) – http://www.tadoc.org/indicator/CORREL.htm
            "CORREL": dict(columns=[self.label_high, self.label_low], parameters=dict(timeperiod=self.timeperiod)),
            # Linear Regression – http://www.tadoc.org/indicator/LINEARREG.htm
            "LINEARREG": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Linear Regression Angle – http://www.tadoc.org/indicator/LINEARREG_ANGLE.htm
            "LINEARREG_ANGLE": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Linear Regression Intercept – http://www.tadoc.org/indicator/LINEARREG_INTERCEPT.htm
            "LINEARREG_INTERCEPT": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Linear Regression Slope – http://www.tadoc.org/indicator/LINEARREG_SLOPE.htm
            "LINEARREG_SLOPE": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod)),
            # Standard Deviation – http://www.tadoc.org/indicator/STDDEV.htm
            "STDDEV": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod, nbdev=self.nbdev)),
            # Variance – http://www.tadoc.org/indicator/VAR.htm
            "VAR": dict(columns=[self.label_close], parameters=dict(timeperiod=self.timeperiod, nbdev=self.nbdev)),
        }

    def __volatility_indicators_arguments(self) -> t.Dict[str, t.Dict]:
        return {
            # Average True Range – http://www.tadoc.org/indicator/ATR.htm
            "ATR": dict(
                columns=[self.label_high, self.label_low, self.label_close], parameters=dict(timeperiod=self.timeperiod)
            ),
            # Normalized Average True RangeNormalized Average True Range – http://www.tadoc.org/indicator/NATR.htm
            "NATR": dict(
                columns=[self.label_high, self.label_low, self.label_close], parameters=dict(timeperiod=self.timeperiod)
            ),
            # True Range – http://www.tadoc.org/indicator/TRANGE.htm
            "TRANGE": dict(columns=[self.label_high, self.label_low, self.label_close], parameters={}),
        }

    def __volume_indicators_arguments(self) -> t.Dict[str, t.Dict]:
        return {
            # Chaikin A/D Line – http://www.tadoc.org/indicator/AD.htm
            "AD": dict(columns=[self.label_high, self.label_low, self.label_close, self.label_volume], parameters={}),
            # Chaikin A/D Oscillator – http://www.tadoc.org/indicator/ADOSC.htm
            "ADOSC": dict(
                columns=[self.label_high, self.label_low, self.label_close, self.label_volume],
                parameters=dict(fastperiod=self.fastperiod, slowperiod=self.slowperiod),
            ),
            # On Balance Volume – http://www.tadoc.org/indicator/OBV.htm
            "OBV": dict(columns=[self.label_close, self.label_volume], parameters={}),
        }
