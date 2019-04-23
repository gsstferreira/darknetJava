package Classes.Lists;

import Classes.KeyValuePair;

public class KeyValuePairElement {

    public final KeyValuePair kvp;
    public KeyValuePairElement next;
    public KeyValuePairElement previous;
    public KeyValuePairList list;

    public KeyValuePairElement(KeyValuePair keyValuePair) {
        this.kvp = keyValuePair;
    }
}
