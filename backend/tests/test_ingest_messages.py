"""Tests for backend.ingest.messages: .z codec, parse_line variants, deep_merge, timestamps."""
from __future__ import annotations

import base64
import json
import zlib

import pytest

from backend.ingest.messages import (
    Msg,
    as_list,
    decode_z,
    deep_merge,
    encode_z,
    format_ts,
    parse_line,
    parse_ts,
    to_raw_line,
)

CAR = {"Entries": [{"Utc": "2024-03-02T15:04:05.123Z", "Cars": {"1": {"Channels": {"0": 11000, "2": 312, "3": 7, "4": 100, "5": 0, "45": 12}}}}]}
POS = {"Position": [{"Timestamp": "2024-03-02T15:04:05.123Z", "Entries": {"1": {"Status": "OnTrack", "X": 1234, "Y": -567, "Z": 10}}}]}
TS = "2024-03-02T15:04:05.123Z"
TS_F = 1709391845.123


def _fastf1_z(obj) -> str:
    """Independent re-implementation of how F1 encodes .z payloads."""
    c = zlib.compressobj(wbits=-15)
    raw = c.compress(json.dumps(obj).encode()) + c.flush()
    return base64.b64encode(raw).decode()


# ----------------------------------------------------------------------------- codec
def test_z_roundtrip_and_interop():
    assert decode_z(encode_z(CAR)) == CAR
    assert decode_z(_fastf1_z(POS)) == POS
    # our encoder must be decodable by the reference decoder (raw deflate, no zlib header)
    raw = base64.b64decode(encode_z(CAR))
    assert json.loads(zlib.decompress(raw, -15)) == CAR


def test_decode_z_tolerates_bom_and_zlib_header():
    c = zlib.compressobj(wbits=-15)
    raw = c.compress(b"\xef\xbb\xbf" + json.dumps(CAR).encode()) + c.flush()
    assert decode_z(base64.b64encode(raw)) == CAR
    framed = base64.b64encode(zlib.compress(json.dumps(CAR).encode()))
    assert decode_z(framed) == CAR


def test_decode_z_garbage_raises():
    with pytest.raises(Exception):
        decode_z("not base64 at all !!!")


# ----------------------------------------------------------------------------- timestamps
@pytest.mark.parametrize(
    "s,expected",
    [
        ("2024-03-02T15:04:05.123Z", TS_F),
        ("2024-03-02T15:04:05Z", 1709391845.0),
        ("2024-03-02T15:04:05.1234567Z", 1709391845.1234567),  # 7 fractional digits like the F1 feed
        ("2024-03-02T15:04:05", 1709391845.0),  # naive == UTC
        ("2024-03-02T17:04:05+02:00", 1709391845.0),
        ("2024-03-02T10:04:05-0500", 1709391845.0),
        ("2024-03-02 15:04:05.5", 1709391845.5),
        (1709391845.5, 1709391845.5),
        (1709391845500, 1709391845.5),  # epoch ms
    ],
)
def test_parse_ts(s, expected):
    assert parse_ts(s) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("bad", [None, "", "garbage", "2024-13-45T99:99:99Z", True, [], {}])
def test_parse_ts_bad(bad):
    assert parse_ts(bad) is None


def test_format_ts_roundtrip():
    assert format_ts(TS_F) == TS
    assert parse_ts(format_ts(1700000000.987)) == pytest.approx(1700000000.987, abs=1e-3)


# ----------------------------------------------------------------------------- parse_line
def test_parse_python_repr_line():
    line = "['TimingData', {'Lines': {'1': {'Position': '1', 'InPit': False, 'Retired': None}}}, '%s']\n" % TS
    m = parse_line(line)
    assert m == Msg("TimingData", {"Lines": {"1": {"Position": "1", "InPit": False, "Retired": None}}}, pytest.approx(TS_F))


def test_parse_z_line_strips_suffix_and_decodes():
    for topic, payload in (("CarData", CAR), ("Position", POS)):
        m = parse_line(str([topic + ".z", encode_z(payload), TS]))
        assert m is not None and m.topic == topic and m.data == payload and m.utc == pytest.approx(TS_F)


def test_parse_json_list_line():
    m = parse_line(json.dumps(["WeatherData", {"AirTemp": "20.1", "Rain": None, "X": True}, TS]))
    assert m.topic == "WeatherData" and m.data["X"] is True and m.data["Rain"] is None


def test_parse_json_lines_dict():
    m = parse_line(json.dumps({"topic": "LapCount", "data": {"CurrentLap": 3}, "utc": TS_F}))
    assert m == Msg("LapCount", {"CurrentLap": 3}, TS_F)
    m = parse_line(json.dumps({"topic": "CarData.z", "data": encode_z(CAR), "utc": TS}))
    assert m.topic == "CarData" and m.data == CAR and m.utc == pytest.approx(TS_F)


def test_parse_completion_message_dump():
    """fastf1 dumps the subscribe reply as ['Topic', '<json string>', ''] -- data given as JSON text."""
    snap = {"1": {"RacingNumber": "1", "Tla": "VER"}}
    m = parse_line(str(["DriverList", json.dumps(snap), ""]), default_utc=42.0)
    assert m.topic == "DriverList" and m.data == snap and m.utc == 42.0
    # .z topic whose JSON string wraps the base64 text
    m = parse_line(str(["CarData.z", json.dumps(encode_z(CAR)), ""]), default_utc=7.0)
    assert m.topic == "CarData" and m.data == CAR
    # snapshot of an ordinary (non-.z) topic that is a JSON *string* of a dict with its own timestamp
    m = parse_line(str(["ExtrapolatedClock", json.dumps({"Utc": TS, "Remaining": "01:00:00", "Extrapolating": True}), ""]))
    assert m.utc == pytest.approx(TS_F)


def test_parse_no_timestamp_uses_default_or_payload():
    assert parse_line("['Heartbeat', {'Utc': '%s'}]" % TS).utc == pytest.approx(TS_F)
    assert parse_line("['LapCount', {'CurrentLap': 1}]", default_utc=9.5).utc == 9.5
    assert parse_line("['LapCount', {'CurrentLap': 1}, '']").utc == 0.0
    # .z payload: fall back to the first entry's own Utc
    m = parse_line(str(["CarData.z", encode_z(CAR), ""]))
    assert m.utc == pytest.approx(TS_F)


def test_parse_bytes_and_whitespace():
    m = parse_line(("  ['LapCount', {'CurrentLap': 2}, '%s']  \r\n" % TS).encode())
    assert m.data == {"CurrentLap": 2}


@pytest.mark.parametrize(
    "garbage",
    [
        "", "   ", "\n", "hello", "not a list", "[", "]", "['TimingData'", "['TimingData', {'a': ", "{bad json",
        "[1, 2, 3]", "['only-topic']", "[None, {}, '']", "['', {}, '']", "{}", '{"topic": 5}', "[[], {}]",
        "['CarData.z', 'AAAA!!!!', '2024-03-02T15:04:05Z']",  # undecodable .z
        "['X', 'not json {', '']",  # string payload that is not JSON
        "['TimingData', {'a': 1}, '2024-03-02T15:04:05Z'] trailing garbage",
        "\x00\x01\x02", "['" + "x" * 10 + "'",
    ],
)
def test_parse_garbage_returns_none(garbage):
    assert parse_line(garbage) is None


def test_to_raw_line_roundtrip():
    m = Msg("CarData", CAR, TS_F)
    enc = to_raw_line(m, encode=True)
    assert enc.startswith("['CarData.z', '") and enc.endswith("%s']" % TS)
    back = parse_line(enc)
    assert back.topic == "CarData" and back.data == CAR and back.utc == pytest.approx(TS_F, abs=1e-3)
    dec = to_raw_line(m, encode=False)
    assert dec.startswith("['CarData', {")
    assert parse_line(dec).data == CAR
    other = Msg("TimingData", {"Lines": {"1": {"InPit": False}}}, TS_F)
    assert parse_line(to_raw_line(other)).data == other.data


# ----------------------------------------------------------------------------- deep_merge
def test_deep_merge_basic():
    dst = {"Lines": {"1": {"Position": "2", "Sectors": {"0": {"Value": "28.1"}}}}}
    out = deep_merge(dst, {"Lines": {"1": {"Position": "1", "Sectors": {"1": {"Value": "30.0"}}}, "2": {"Position": "2"}}})
    assert out is dst
    assert dst == {"Lines": {"1": {"Position": "1", "Sectors": {"0": {"Value": "28.1"}, "1": {"Value": "30.0"}}},
                             "2": {"Position": "2"}}}


def test_deep_merge_replaces_non_dicts_and_none():
    dst = {"a": {"x": 1}, "b": [1, 2], "c": 5}
    deep_merge(dst, {"a": 7, "b": [9], "c": None})
    assert dst == {"a": 7, "b": [9], "c": None}


def test_deep_merge_does_not_alias_patch():
    patch = {"Lines": {"1": {"Sectors": {"0": {"Value": "1"}}}}}
    dst = deep_merge({}, patch)
    patch["Lines"]["1"]["Sectors"]["0"]["Value"] = "mutated"
    assert dst["Lines"]["1"]["Sectors"]["0"]["Value"] == "1"
    dst2 = {}
    deep_merge(dst2, patch)
    dst2["Lines"]["1"]["x"] = 1
    assert "x" not in patch["Lines"]["1"]


def test_deep_merge_list_patched_by_index_dict():
    dst = {"Messages": [{"Message": "a"}]}
    deep_merge(dst, {"Messages": {"1": {"Message": "b"}, "0": {"Lap": 3}}})
    assert dst["Messages"] == [{"Message": "a", "Lap": 3}, {"Message": "b"}]
    deep_merge(dst, {"Messages": {"4": {"Message": "e"}}})
    assert len(dst["Messages"]) == 5 and dst["Messages"][3] is None and dst["Messages"][4] == {"Message": "e"}
    # a full list replaces
    deep_merge(dst, {"Messages": [{"Message": "z"}]})
    assert dst["Messages"] == [{"Message": "z"}]


def test_deep_merge_non_dict_destination():
    assert deep_merge(None, {"a": 1}) == {"a": 1}
    assert deep_merge(3, {"a": {"b": 1}}) == {"a": {"b": 1}}


def test_deep_merge_dict_keyed_by_index_into_dict():
    dst = {"Stints": {"0": {"Compound": "SOFT", "TotalLaps": 3}}}
    deep_merge(dst, {"Stints": {"0": {"TotalLaps": 4}, "1": {"Compound": "HARD"}}})
    assert dst["Stints"] == {"0": {"Compound": "SOFT", "TotalLaps": 4}, "1": {"Compound": "HARD"}}


def test_as_list():
    assert as_list(None) == []
    assert as_list([1, 2]) == [1, 2]
    assert as_list({"10": "c", "2": "b", "0": "a"}) == ["a", "b", "c"]
    assert as_list({"x": 1}) == [1]
    assert as_list(5) == []
