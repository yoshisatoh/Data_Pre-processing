import matplotlib.pyplot as plt

#from lxml import html    #yoshisatoh updated
import requests
#from xml.etree import ElementTree    #yoshisatoh updated

from ratelimit import limits, sleep_and_retry

#yoshisatoh updated
headers = {'accept': 'application/xml;q=0.9, */*;q=0.8'}
'''
# To read an web page as an xml file, not an html file, try the following example:

url = 'https://www.treasurydirect.gov/TA_WS/securities/announced/rss'
headers = {'accept': 'application/xml;q=0.9, */*;q=0.8'}
response = requests.get(url, headers=headers, verify=False)
print(response.text)
'''


class SecAPI(object):
    SEC_CALL_LIMIT = {'calls': 10, 'seconds': 1}
    #
    @staticmethod
    @sleep_and_retry
    # Dividing the call limit by half to avoid coming close to the limit
    @limits(calls=SEC_CALL_LIMIT['calls'] / 2, period=SEC_CALL_LIMIT['seconds'])
    def _call_sec(url):
        #
        #return requests.get(url)
        #print(requests.get(url, verify=False))    #yoshisatoh updated
        #print(requests.get(url, verify=False).headers['content-type'])    #yoshisatoh updated
        #return requests.get(url, verify=False)    #yoshisatoh updated
        #print(requests.get(url, headers=headers, verify=False))    #yoshisatoh updated
        return requests.get(url, headers=headers, verify=False)    #yoshisatoh updated
        #
        #response =  requests.get(url, verify=False)    #yoshisatoh updated
        #return ElementTree.fromstring(response.content)
    #
    def get(self, url):
        return self._call_sec(url).text


def print_ten_k_data(ten_k_data, fields, field_length_limit=50):
    indentation = '  '

    print('[')
    for ten_k in ten_k_data:
        print_statement = '{}{{'.format(indentation)
        for field in fields:
            value = str(ten_k[field])

            # Show return lines in output
            if isinstance(value, str):
                value_str = '\'{}\''.format(value.replace('\n', '\\n'))
            else:
                value_str = str(value)

            # Cut off the string if it gets too long
            if len(value_str) > field_length_limit:
                value_str = value_str[:field_length_limit] + '...'

            print_statement += '\n{}{}: {}'.format(indentation * 2, field, value_str)

        print_statement += '},'
        print(print_statement)
    print(']')


def plot_similarities(similarities_list, dates, title, labels):
    assert len(similarities_list) == len(labels)

    plt.figure(1, figsize=(10, 7))
    for similarities, label in zip(similarities_list, labels):
        plt.title(title)
        plt.plot(dates, similarities, label=label)
        plt.legend()
        plt.xticks(rotation=90)

    plt.show()
