from bs4 import BeautifulSoup

dataset = np.load('../dataset/intention_annotation_784.npy', allow_pickle = True)

for post in dataset:
	bs_text = BeautifulSoup(post['description_raw'])
	post['description'] = bs_text.get_text()

    cb_list = []
    cbs = bs_text.select("pre > code")
    if len(cbs)!=0:
        for cb in cbs:
            cb_list.append(cb.get_text())
            cb.clear()
	post['code'] = cb_list