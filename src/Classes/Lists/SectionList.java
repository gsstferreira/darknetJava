package Classes.Lists;

import Classes.Section;

public class SectionList {

    private int totalSize;
    private SectionElement head;
    private SectionElement tail;

    public int size() {
        return totalSize;
    }

    public SectionList() {
        this.totalSize = 0;
        this.head = new SectionElement(new Section());
    }

    public void add(Section s) {

        SectionElement se = new SectionElement(s);

        if(totalSize == 0) {
            head.next = se;
            tail = se;
        }
        else {
            tail.next = se;
            tail = se;
        }
        totalSize++;
    }

    public SectionElement get(int index) {

        SectionElement s = head;

        for(int i = 0; i < index; i++) {
            s = s.next;
        }

        return s.next;
    }

    public SectionElement remove(int index) {

        SectionElement s = head;
        SectionElement s1 = head;
        for(int i = 0; i < index; i++) {
            s1 = s;
            s = s.next;
        }
        s1.next = s.next.next;
        totalSize--;

        return s.next;
    }

    public SectionElement pop() {

        if(totalSize == 0) {
            return null;
        }
        else {
            SectionElement se = head.next;
            head.next = head.next.next;
            totalSize--;

            return se;
        }
    }
}
