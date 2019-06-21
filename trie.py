"""
@author: 徐恩
"""

class TrieTree:

    def __init__(self, src_file, merge_tags=True):
        self.data = {}
        self.endF = u'$'
        self.merge_tags = merge_tags
        self.all_tags = set([])
        with open(src_file, 'r') as f:
            for line in f:
                self.insert(line.strip())

    def insert(self, pre, tag_iter=[]):
        if not pre:
            return
        wtree = self.data
        for word in pre:
            if word == self.endF:
                word = u'\\' + word
            if word not in wtree:
                wtree[word] = {}
            wtree = wtree[word]

        if self.endF in wtree:
            endFlag = wtree[self.endF]
            for tag in tag_iter:
                if tag in endFlag:  # replace the old element
                    endFlag.remove(tag)
                endFlag.add(tag)
        else:
            wtree[self.endF] = set(tag_iter)
        if self.merge_tags:
            for tag in tag_iter:
                self.all_tags.add(tag)

    def hasPre(self, pre, tag_list=[]):
        if not pre:
            return (False, None)

        tag_list = set(tag_list)
        if tag_list and self.merge_tags:
            if not (self.all_tags & tag_list):
                return (False, None)

        subtree = self.data
        lnum = 0
        for word in pre:
            if word == self.endF:
                word = u'\\' + word
            if word in subtree:
                lnum += 1
                subtree = subtree[word]
            else:
                break
        if lnum != len(pre):
            return (False, None)

        if self.endF in subtree:
            flags = subtree[self.endF]
            if tag_list and (not flags & tag_list):
                return (False, None)
            return (True, flags)
        return (False, None)

    def __str_dis__(self, deep, node, bufferP, head):
        if deep > 0:
            times = deep
            while times > 0:
                bufferP.append(u' |')
                times -= 1
            bufferP.append(u'-')
        bufferP.append(head)
        if self.endF in node:
            bufferP.append(u'*')
        bufferP.append(u'\n')
        deep += 1
        for (hd, nod) in node.items():
            if hd == self.endF:
                continue
            self.__str_dis__(deep, nod, bufferP, hd)

    def __str__(self):
        bufferP = []
        self.__str_dis__(0, self.data, bufferP, 'trie')
        return ''.join(bufferP)

    def removeTag(self, tag_list):
        tag_list = set(tag_list)
        if self.merge_tags:
            left_tag = self.all_tags & tag_list
            if not left_tag:
                return
            for tag in left_tag:
                self.all_tags.remove(tag)

        del_path = []
        del_path.append([self.data, [], True])
        while del_path:
            info = del_path[-1]
            if not info[1]:
                if info[2]:
                    for (hd, nod) in info[0].items():
                        if hd == self.endF:
                            tr = nod & tag_list
                            if tr:
                                [nod.remove(fl) for fl in tr]
                            if len(nod) < 1:
                                del info[0][self.endF]
                        else:
                            info[1].append(nod)
                    info[2] = False
                else:
                    dels_node = []
                    for (hd, nod) in info[0].items():
                        if not nod:
                            dels_node.append(hd)
                    for dhd in dels_node:
                        del info[0][dhd]
                    del_path.pop()
            else:
                del_path.append([info[1].pop(), [], True])

    def removePre(self, pre, tag_list=[]):
        if not pre:
            return
        tag_list = set(tag_list)
        if tag_list and self.merge_tags:
            if not self.all_tags & tag_list:
                return

        subtree = self.data
        rk = []
        for word in pre:
            if word == self.endF:
                word = u'\\' + word
            if word in subtree:
                rk.append(subtree)
                subtree = subtree[word]
            else:
                break
        if len(rk) != len(pre):
            return

        if self.endF in subtree:
            flags = subtree[self.endF]
            if tag_list:
                tr = flags & tag_list
                if not tr: return
                for tag in tr:
                    flags.remove(tag)
                if flags:
                    return

            del subtree[self.endF]
            lk = zip(rk, pre)
            lk.reverse()
            for (li, wd) in lk:
                if not li[wd]: del li[wd]

    def __dump_result__(self, result):
        max_l = len(result)
        if max_l <= 1:
            return result
        new_result = []
        i = -1
        while i < max_l - 2:
            i += 1
            p_r = result[i]
            not_use = False
            for j in range(i + 1, max_l):
                n_r = result[j]
                if p_r[2] >= n_r[2] and p_r[3] <= n_r[3]:
                    not_use = True
                    break
            if not not_use:
                new_result.append(p_r)
        new_result.append(result[max_l - 1])
        return new_result

    def cut(self, pre):
        temp = []
        cut = self.contains(pre, dump=True)
        for word in cut:
            temp.append(''.join(word[0]))
        return temp

    def contains(self, pre, tag_list=[], dump=False):
        """
        check the pre list in the pres,return
        [(hotword_list,tag_list,start,end)]
        """
        pre = pre.lower()
        if not pre:
            return []
        tag_list = set(tag_list)
        if tag_list and self.merge_tags:
            if not self.all_tags & tag_list:
                return []

        use_list = []
        result = []
        for (idx, word) in enumerate(pre):
            word_o = word
            if word_o == self.endF:
                word_o = u'\\' + word_o

            for index in range(len(use_list) - 1, -1, -1):
                ptrlist = use_list[index]
                if self.endF in ptrlist[0]:
                    if tag_list:
                        rt = ptrlist[0][self.endF] & tag_list
                        if rt: result.append((list(ptrlist[1]), list(rt), ptrlist[2], idx - 1))
                    else:
                        result.append((list(ptrlist[1]), list(ptrlist[0][self.endF]), ptrlist[2], idx - 1))
                if word_o in ptrlist[0]:
                    ptrlist[0] = ptrlist[0][word_o]
                    ptrlist[1].append(word)
                    continue
                del use_list[index]
            if word_o in self.data:
                use_list.append([self.data[word_o], [word], idx])

        for ptrlist in reversed(use_list):
            if self.endF in ptrlist[0]:
                if tag_list:
                    rt = ptrlist[0][self.endF] & tag_list
                    if rt: result.append((ptrlist[1], list(rt), ptrlist[2], idx))
                else:
                    result.append((ptrlist[1], list(ptrlist[0][self.endF]), ptrlist[2], idx))

        if dump:
            return self.__dump_result__(result)

        return result
