import xml.etree.ElementTree as ET
import json
import html
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup

#Used to split WN-Salience-articles-v0.xml into WN-Salience-articles-train.xml and  WN-Salience-articles-test.xml and move splits into ../raw_data
#Also used to get rid of html characters like &pound; from the original XML
def split_wn_salience_xml_train_val_test():

    # Path to the raw XML file
    file_path = '../raw_data/WN-Salience-articles-v0.xml'

    # Read the XML content into a string
    with open(file_path, 'r',  encoding='utf-8') as file:
        xml_data = file.read()
    
    #Need to map all standalone '&' to &amp; before passing to ET.fromstring or else ET.fromstring will crash
    #Also need to unescape html characters &pound; first, then convert all remaining & to &amp;
    root = ET.fromstring(html.unescape(xml_data).replace('&', '&amp;'))

    #Lists containing train, val, and test articles
    train_article_pages = []
    val_article_pages = []
    test_article_pages = []

    # Access elements of XML ElementTree object which are articles
    for article_page in root:
        for child in article_page:
            if child.tag == "ap-date":
                #If article is from January 1 to October 31 send to train/val
                #YYYY-MM-DD is date format
                if int(child.text[5:7]) < 11:
                    #If from 2013 and before, send to train
                    if int(child.text[:4]) < 2014:
                        train_article_pages.append(article_page)
                    #Else from 2014 and later, send to val
                    else:
                        val_article_pages.append(article_page)
                #If article from November 1 to December 31 send to test
                else:
                    test_article_pages.append(article_page)

    #Printing number of articles in each split
    print(f'number of train articles: {len(train_article_pages)}')
    print(f'number of val articles: {len(val_article_pages)}')
    print(f'number of test articles: {len(test_article_pages)}')

    # Create a root element for train article pages XML
    train_root = ET.Element("root")

    # Append all elements to the root
    for elem in train_article_pages:
        train_root.append(elem)

    # Create a tree and write to a file
    train_tree = ET.ElementTree(train_root)
    with open("../raw_data/WN-Salience-articles-train.xml", "wb") as f:
        train_tree.write(f, encoding="utf-8")

    val_root = ET.Element("root")
    for elem in val_article_pages:
        val_root.append(elem)

    val_tree = ET.ElementTree(val_root)
    with open("../raw_data/WN-Salience-articles-val.xml", "wb") as f:
        val_tree.write(f, encoding="utf-8")

    test_root = ET.Element("root")
    for elem in test_article_pages:
        test_root.append(elem)

    # Create a tree and write to a file
    test_tree = ET.ElementTree(test_root)
    with open("../raw_data/WN-Salience-articles-test.xml", "wb") as f:
        test_tree.write(f, encoding="utf-8")
          
def train_val_test_xml_to_json():
    #Run the script for train, validation, and test sets
    for train_val_test in ["train", "val", "test"]:

        xml_data = ""
        #used to store information for all articles, so each entry will be a dictionary with information for current article
        all_article_data = []

        # Path to the XML file
        file_path = f'../raw_data/WN-Salience-articles-{train_val_test}.xml'

        # Read the XML content into a string
        with open(file_path, 'r', encoding='utf-8') as file:
            xml_data = file.read()

        # Parse from string
        root = ET.fromstring(xml_data)

        #Looping through articles
        for article_page in root:
            cur_article_entities = []
            #Will be a dictionary containing information for the article
            article_data = {"text": ""}
            #Used to keep track of article paragraph lengths for the current article for begin and end offset calculations
            article_paragraph_lengths = []
            #Looping through information for current article
            for child in article_page:
                if child.tag == "ap-date":
                    article_data["date"] = child.text
                elif child.tag == "ap-title":
                    article_data["title"] = child.text
                #If on a pagragraph for the article
                elif child.tag == "paragraph":
                    cur_paragraph_text = ""
                    #key is mention, value is most recent start index
                    paragraph_entity_offsets = {}
                    #Looping through current paragraph tags
                    for paragraph_child in child:
                        #If current tag is for the content/text of the body paragraph
                        if paragraph_child.tag == "content":
                            #Needed to convert &amp; to & from the XML
                            decoded = html.unescape(paragraph_child.text)
                            #Append to article body string for current article
                            article_data["text"] = article_data["text"] + decoded + " "
                            #Adding space for the start of the next paragraph
                            cur_paragraph_text = decoded + " "
                            #Keeping track of article paragraph lengths
                            article_paragraph_lengths.append(len(cur_paragraph_text))
                        #contains information for a salient (or non-salient) entity, gives a wikipedia page link too  
                        if paragraph_child.tag == "annotation":
                            entity_title = None
                            entity_salience = None
                            begin_offset = None
                            end_offset = None
                            url = None
                            #Will be false if entity_title is None or url is None or if we've already seen the wiki URL in ground truth for this article
                            valid_entity = True
                            #Loop through tags for annotation
                            for annotation_child in paragraph_child:
                                if annotation_child.tag == "mention":
                                    entity_title = annotation_child.text
                                if annotation_child.tag == "salience":
                                    entity_salience = annotation_child.text 
                                if annotation_child.tag == "url":
                                    url = annotation_child.text 
                                if annotation_child.tag == "beginOffset":
                                    #At this point if we do not have the entity title or do not have the wiki URL for the mention, then break and move onto next annotation
                                    if entity_title is None or url is None:
                                        valid_entity = False
                                        break
                                    #Or if we have already seen the same wiki URL for this article before, also break because we only care about first occurence of the entity in the article
                                    if any(entity["url"] == url for entity in cur_article_entities):
                                        valid_entity = False
                                        break
                                    #if first time seeing mention in current paragraph
                                    if entity_title not in paragraph_entity_offsets:
                                        #use python find function to find starting offset (into whole article body)
                                        begin_offset = str(cur_paragraph_text.find(entity_title) + sum(article_paragraph_lengths[:-1]))
                                        paragraph_entity_offsets[entity_title] = cur_paragraph_text.find(entity_title)
                                    #If not the first time seeing the mention in the current paragraph
                                    else:
                                        #Get most recent begin offset of the mention from the dictionary
                                        most_recent_begin_offset = paragraph_entity_offsets[entity_title]
                                        #find next occurence of the mention after the end index of most recent one
                                        new_begin_offset = str(cur_paragraph_text.find(entity_title, most_recent_begin_offset + len(entity_title)) + sum(article_paragraph_lengths[:-1]))
                                        #Update most recent mention begin offset into current paragraph
                                        paragraph_entity_offsets[entity_title] = cur_paragraph_text.find(entity_title, most_recent_begin_offset + len(entity_title))
                                        begin_offset = new_begin_offset
                                if annotation_child.tag == "endOffset":
                                    #already have behgin_offset from above
                                    end_offset = str(int(begin_offset) + len(entity_title))
                            #Only append if entity has a wiki URL and not an empty mention tag
                            if valid_entity:
                                cur_article_entities.append({"entity title": entity_title, "entity salience": entity_salience, "begin offset": begin_offset, "end offset": end_offset, "url": url})		
            article_data["entities"] = cur_article_entities
            #To deal with stray space at end of article
            article_data["text"] = article_data["text"][:-1]
            all_article_data.append(article_data)

        # Write the article data array to a JSON file
        with open(f'../raw_data/WNS_{train_val_test}.json', 'w', encoding="utf-8") as json_file:
            json.dump(all_article_data, json_file, indent=4, ensure_ascii=False)

#Take article_info_train_fixed.json and article_info_test_fixed.json and convert to CSV while ALSO making validation split out of train
def json_to_csv():

    kb_path = '/work/pi_wenlongzhao_umass_edu/8/aranade/696-detecting-salient-entities/src/prep_kb/wiki_id_metadata.json'

    #Load in knowledge base
    with open(kb_path, 'r') as file:
        kb = json.load(file)
        print('kb loaded')

    for train_val_test in ["test"]:
        #Read to pandas df from json
        train_val_test_df = pd.read_json(f'../raw_data/WNS_{train_val_test}.json')
        #Contains rows of final csv
        flattened_rows = []
        for _, article in train_val_test_df.iterrows():
            for entity in article['entities']:
                #skipping non salient entities for test set
                if train_val_test == "test" and entity['entity salience'] == "0":
                    continue
                #attempt to get valid id
                fetched_wiki_page_id = fetch_wiki_page_id(entity["url"])
                #If no wiki ID exists for entity or wiki ID is not in knowledge base then continue
                if not fetched_wiki_page_id or str(fetched_wiki_page_id) not in kb:
                    #print(f'not found: {entity["url"]}')
                    continue
                flattened_rows.append({
                    "text": article['text'],
                    "date": article['date'],
                    "title": article['title'],
                    "entity title": entity['entity title'],
                    "entity salience": int(entity['entity salience']),
                    "offsets": (int(entity['begin offset']), int(entity['end offset'])),
                    "url": entity['url'],
                    "wiki_ID": fetched_wiki_page_id
                })

        # Create the flattened DataFrame
        train_val_test_df = pd.DataFrame(flattened_rows)
        #Used to drop duplicates and reset indicies
        train_val_test_df = train_val_test_df.drop_duplicates().reset_index(drop=True)

        #If on test set, drop offsets and non salient entities
        if train_val_test == "test":
            #Only keeping salient entities
            train_val_test_df[train_val_test_df["entity salience"] == 1].drop("offsets", axis=1).to_csv("../splits/WNS_test_KB.csv", index=False)
        
        #If on train or val, keep offsets and non salient entities
        else:
            if train_val_test == "train":
                train_val_test_df.to_csv("../splits/WNS_train_KB.csv", index=False)
            else:
                train_val_test_df.to_csv("../splits/WNS_val_KB.csv", index=False)

#wiki_url will either be "https://en.wikipedia.org/wiki/{entity}" or just "/wiki/{entity}"
def fetch_wiki_page_id(wiki_url):

    wiki_page_title_start_offset = wiki_url.find('/wiki/') + 6
    wiki_page_title = wiki_url[wiki_page_title_start_offset:]

    headers = {
        'User-Agent': 'Your-App-Name/1.0 (your-email@example.com)'
    }

    response = requests.get(f'https://en.wikipedia.org/w/index.php?title={wiki_page_title}&action=info', headers=headers)

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')    

    #Immediately check for a redirect
    redirect_row = soup.find('tr', {'id': 'mw-pageinfo-redirectsto'})

    #Means redirect row exists so go there instead to try and get page id
    if redirect_row:

        #Now redirect href is something like /w/index.php?title=Atmosphere_of_Earth&action=info" 
        new_page_title = redirect_row.find_all('td')[1].find_all('a')[0].get('href')[6:]

        response = requests.get(f'https://en.wikipedia.org/w/index.php?title={new_page_title}&action=info', headers=headers)

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        page_id_row = soup.find('tr', {'id': 'mw-pageinfo-article-id'})

        # Extract the Page ID either from the new redirect info page
        if page_id_row:
            page_id = page_id_row.find_all('td')[1].text.strip()
            if int(page_id) == 0:
                # print(f'Page ID not found for {wiki_url}')
                return None
            else:
                return(int(page_id))
        #If no page id on the redirect page, return None
        else:
            return None

    # If redicrect row does not exist means that only possible valid wiki ID is on this page
    else:
        page_id_row = soup.find('tr', {'id': 'mw-pageinfo-article-id'})

        #If page id row doesn't exist on the original page then return none
        if not page_id_row:
            return None

        else:
            page_id = page_id_row.find_all('td')[1].text.strip()
            if int(page_id) == 0:
                return None
            else:
                return(int(page_id))

#Used for adding Wikidata Q IDs to WN-Salience ground truth by mapping Wikipedia page IDs to respective Q ID 
def add_q_ids():

    headers = {
        "User-Agent": "MyWikidataMapper/1.0 (your_email@example.com)"  # <-- required!
    }

    for train_val_test in ["test"]:

        wns_df = pd.read_csv(f"/work/pi_wenlongzhao_umass_edu/8/696-detecting-salient-entities/data/WN_csv_new/WNS_{train_val_test}_KB.csv")
        wiki_ids = wns_df["wiki_ID"].to_list()

        for idx, wiki_id in enumerate(wiki_ids):
            url = f"https://en.wikipedia.org/w/api.php?action=query&pageids={wiki_id}&prop=pageprops&format=json"

            response = requests.get(url, headers=headers)
            data = response.json()

            qid = data["query"]["pages"][str(wiki_id)]["pageprops"]["wikibase_item"]
            #if not mapped correctly for some reason
            if not qid:
                mapped_q_ids.append(None)
            mapped_q_ids.append(qid)

        test_df["Q_ID"] = mapped_q_ids
        test_df.to_csv(f"/work/pi_wenlongzhao_umass_edu/8/james/salient-entity-linking-with-llm/data/wn_salience/splits/WNS_{train_val_test}_QID_KB.csv")


if __name__ == "__main__":

    split_wn_salience_xml_train_val_test()
    train_val_test_xml_to_json()
    json_to_csv()
    add_q_ids()