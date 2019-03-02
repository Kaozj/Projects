import json
import urllib
from urllib.parse import urlparse
import httplib2 as http #External library 

if __name__=="__main__":
    #Authentication parameters
    headers = { 'AccountKey' : 'Klo8M9spSBqnTw8dZ3Im2Q==',
    'accept' : 'application/json'} #this is by default

    #API parameters
    uri = 'http://datamall2.mytransport.sg/' #Resource URL 
    path = '/ltaodataservice/BusServices'

    #Build query string & specify type of API call
    target = urlparse(uri + path) 
    print (target.geturl())
    method = 'GET'
    body = ''
    
    #Get handle to http
    h = http.Http()
        
    #Obtain results
    response, content = h.request(
        target.geturl(),
        method,
        body,
        headers)

    #Parse JSON to print
    jsonObj = json.loads(content)
    print (json.dumps(jsonObj, sort_keys=True, indent=4))

    #Save result to file
    with open("busstop83139.json","w") as outfile:
        #Saving jsonObj["d"]
        json.dump(jsonObj, outfile, sort_keys=True, indent=4,ensure_ascii=False)