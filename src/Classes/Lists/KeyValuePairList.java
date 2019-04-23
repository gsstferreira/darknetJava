package Classes.Lists;

import Classes.KeyValuePair;

public class KeyValuePairList {

    private int totalSize;
    private final KeyValuePairElement head;
    private KeyValuePairElement tail;

    public KeyValuePairList() {
        this.totalSize = 0;
        this.head = new KeyValuePairElement(new KeyValuePair("",""));
    }

    public void add(KeyValuePair s) {

        KeyValuePairElement se = new KeyValuePairElement(s);
        se.list = this;

        if(totalSize == 0) {
            head.next = se;
            se.previous = head;
            tail = se;
        }
        else {
            tail.next = se;
            se.previous = tail;
            tail = se;
        }
        totalSize++;
    }

    public KeyValuePairElement get(int index) {

        KeyValuePairElement s = head;

        for(int i = 0; i < index; i++) {
            s = s.next;
        }

        return s.next;
    }

    public KeyValuePairElement remove(int index) {

        KeyValuePairElement s = head;
        KeyValuePairElement s1 = head;
        for(int i = 0; i < index; i++) {
            s1 = s;
            s = s.next;
        }
        s1.next = s.next.next;
        totalSize--;

        return s.next;
    }

    public void remove(KeyValuePairElement kvpe) {

        if(kvpe != null && kvpe.list == this) {
            kvpe.previous.next = kvpe.next;
            totalSize--;
        }
    }

    public KeyValuePairElement pop() {

        if(totalSize == 0) {
            return null;
        }
        else {
            KeyValuePairElement se = head.next;
            head.next = head.next.next;
            totalSize--;

            return se;
        }
    }

    public KeyValuePairElement getHead() {
        return this.head;
    }
}
