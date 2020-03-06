import json
import requests


class BingSpellCheck(object):
    def __init__(self, key):
        self.api_key = key
        self.endpoint = "https://api.cognitive.microsoft.com/bing/v7.0/SpellCheck"

    def spell_check(self, text, mode='spell'):
        data = {'text': text}

        params = {
            'mkt': 'en-us',
            'mode': mode
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Ocp-Apim-Subscription-Key': self.api_key,
        }
        response = requests.post(
            self.endpoint, headers=headers, params=params, data=data)

        corrected_spells = response.json()

        flaggedTokens = corrected_spells['flaggedTokens']

        for flagged in flaggedTokens:
            text = text.replace(
                flagged['token'], flagged['suggestions'][0]['suggestion'])

        return text
