import requests

nomics_url = 'https://api.nomics.com/v1/'
nomics_api_key = 'b9c8f7b7c3efffc527ff3a546f25ed63'
bittrex_url = 'https://api.bittrex.com/v3/'
bittrex_api_key = ''
# bittrex
# Key
# 3f83843449e74165b43f990e772c8e42
# Secret
# b1a04e561c684829af12a87a626e5841
def get_nomics (endpoint, params):
    # Use & for seperating params
    response = requests.get(nomics_url + endpoint + '?key=' + nomics_api_key + params)
    return response
def get_bittrex (endpoint, params):
    response = requests.get(bittrex_url + endpoint + params)
    return response