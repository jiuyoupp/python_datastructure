class Node(object):
    """结点"""
    def __init__(self, item):
        self.elem = item
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self, newdata):
        self.data = newdata

    def setNext(self, newnext):
        self.next = newnext


class SingleCycleLinkList(object):
    """单向循环链表"""
    def __init__(self, node=None):
        self.head = node
        node.next = node

    def is_empty(self):
        # 链表是否为空
        return self.head == None

    def travel_end(self, node):
        # 遍历到尾节点
        while cur.next != self.head:
            cur = cur.next

    def append(self, item):
        # 尾插法
        node = Node(item)
        if self.is_empty():
            self.head = node
            node.next = node
        else:
            cur = self.head
            while cur.next != self.head:
                cur = cur.next
            cur.next = node
            node.next = self.head

    def remove(self, item):
        # 删除结点
        if self.is_empty():
            return
        cur = self.head
        pre = None
        while cur.next != self.head:
            if cur.elem == item:
                # 删除位置在头节点且多于一个元素的情况
                if cur == self.head:
                    # 先找尾节点
                    rear = self.head
                    while rear.next != self.head:
                        rear = rear.next
                    rear.next = cur.next
                    self.head = cur.next
                else:
                    # 中间结点和尾结点的情况
                    pre.next = cur.next
                return
            else:
                pre = cur
                cur = cur.next

        # 退出循环指向尾结点
        if cur.elem == item:
            # 一个元素的情况删掉后变为空节点  防止pre为none导致出错
            if cur == self.head:
                self.head = None
                # 尾节点元素的情况
            else:
                pre.next = self.head
                # pre.next = cur.next

    def judgement(self, vital):
        cur = self.head
        count = 1
        while cur != cur.next :
            cur = cur.next
            count += 1
            if count == vital:
                self.remove(cur.elem)
                print("%d-->" % cur.elem, end="")
                count = 0
        print(cur.elem)


def joseph(num, vital):
    sll = SingleCycleLinkList()
    for i in range(1, num+1):
        sll.append(i)
    sll.judgement(vital)


joseph(41, 3)