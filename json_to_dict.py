import json

class Json(object):
    """Class to load and decode sentence from json files."""
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.__dict__ = json.load(f)
    def load_messages(self):
        """Return a dictionnary with senders as keys and good message encoding."""
        dict_message = dict()
        for m in self.messages: # iterate through message list / m is a dictionnary
            if m['type'] == 'Generic':
                if m.get('content') != None:
                    content = m.get('content').encode('latin1').decode('utf8')
                    sender_name = m.get('sender_name').encode('latin1').decode('utf8')
                    list_message = dict_message.get(sender_name) # Message list of sender
                    if list_message == None:
                        dict_message[sender_name] = [content]
                    else:
                        list_message.append(content)
                        dict_message[sender_name] = list_message
        return dict_message