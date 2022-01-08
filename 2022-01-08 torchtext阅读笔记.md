关于tokenizer: 把原始字符串切割成多个单独元素的工具，可以类比为中文的分词。

关于build_vocab_from_iterator：建立词表，是把语料中的词按照词频排序，并赋予每个词index。

关于DataLoader：它类似一个map，接受一个迭代器，返回一个迭代器(类似迭代器但具体的数据结构不同，可以用iter(dataloader)转换)，可以看作是一个高级数据管道。

``` python
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

train_iter = AG_NEWS(split='train')
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)
```

此例中，接收train.iter迭代器，里面存放的是原始数据，其中只有标签和原始语料。
每次读取batch_size个数据，在collate_batch中定义和处理返回的数据。

关于embeddingbag:了解中
