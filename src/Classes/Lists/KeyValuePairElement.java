package Classes.Lists;

import Classes.KeyValuePair;

public class KeyValuePairElement {

    public KeyValuePair kvp;
    public KeyValuePairElement next;
    public KeyValuePairElement previous;
    public KeyValuePairList list;

    public KeyValuePairElement(KeyValuePair keyValuePair) {
        this.kvp = keyValuePair;
    }
}
