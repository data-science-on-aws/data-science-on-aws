import pytest
from handler import predict_answer
import json


test_events = {
    "body": '{"question": "Who has the most covid-19 deaths?", "context":"The US has passed the peak on new coronavirus cases,President Donald Trump said and predicted that some states would reopen this month. The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world."}'
}


def test_handler():
    res = predict_answer(test_events, '')
    assert json.loads(res['body']) == {'answer': 'the us'}
