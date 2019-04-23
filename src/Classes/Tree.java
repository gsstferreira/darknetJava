package Classes;

import Classes.Buffers.FloatBuffer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Tree {

    public int[] leaf;
    public int n;
    public int[] parent;
    public int[] child;
    public int[] group;
    public String[] name;

    public int groups;
    public int[] groupSize;
    public int[] groupOffset;

//    public void changeLeaves(String leafList) {
//
//        try {
//            String[] leaves = (String[]) (Data.getPaths(leafList)).toArray();
//            int _n = leaves.length;
//
//            for(int i = 0; i < this.n; i++) {
//                this.leaf[i] = 0;
//                for(int j = 0; j < _n; j++) {
//                    if(this.name[i].equals(leaves[j])) {
//                        this.leaf[i] = 1;
//                    }
//                }
//            }
//        }
//        catch (Exception e) {
//            System.out.println("Error trying to change tree leaves");
//        }
//    }
//
//    public float getHierarchyprobability(float[] x, int c, int stride) {
//
//        float p = 1;
//        while(c >= 0) {
//            p *= x[c*stride];
//            c = this.parent[c];
//        }
//        return p;
//    }

    public float getHierarchyprobability(FloatBuffer x, int c, int stride) {

        float p = 1;
        while(c >= 0) {
            p *= x.get(c*stride);
            c = this.parent[c];
        }
        return p;
    }

//    public void hierarchyPredictions(float[] predictions, int n, boolean onlyLeaves, int stride) {
//
//        for(int i = 0; i < n; i++) {
//
//            int parent = this.parent[i];
//            if(parent >= 0) {
//                predictions[i*stride] *= predictions[parent*stride];
//            }
//        }
//        if(onlyLeaves) {
//            for(int i = 0; i < n; i++) {
//
//                if(this.leaf[i] != 0) {
//                    predictions[i*stride] = 0;
//                }
//            }
//        }
//    }

    public void hierarchyPredictions(FloatBuffer predictions, int n, boolean onlyLeaves, int stride) {

        for(int i = 0; i < n; i++) {

            int parent = this.parent[i];
            if(parent >= 0) {

                predictions.put(i*stride,predictions.get(i*stride) * predictions.get(parent*stride));
            }
        }
        if(onlyLeaves) {
            for(int i = 0; i < n; i++) {

                if(this.leaf[i] != 0) {

                    predictions.put(i*stride,0);
                }
            }
        }
    }

//    public int hierarchyTopPredictions(float[] predictions, float thresh, int stride) {
//
//        float p = 1;
//        int group = 0;
//
//        while(true) {
//            float max = 0;
//            int max_i = 0;
//
//            for(int i = 0; i <this.groupSize[group]; i++) {
//
//                int index = i + this.groupOffset[group];
//                float val = predictions[(i + this.groupOffset[group])*stride];
//
//                if(val > max) {
//                    max_i = index;
//                    max = val;
//                }
//            }
//
//            if(p*max > thresh) {
//                p *= max;
//                group = this.child[max_i];
//
//                if(this.child[max_i] < 0) {
//                    return max_i;
//                }
//            }
//            else if(group == 0) {
//                return max_i;
//            }
//            else {
//                return this.parent[this.groupOffset[group]];
//            }
//        }
//    }

    public int hierarchyTopPredictions(FloatBuffer predictions, float thresh, int stride) {

        float p = 1;
        int group = 0;

        while(true) {
            float max = 0;
            int max_i = 0;

            for(int i = 0; i <this.groupSize[group]; i++) {

                int index = i + this.groupOffset[group];
                float val = predictions.get((i + this.groupOffset[group])*stride);

                if(val > max) {
                    max_i = index;
                    max = val;
                }
            }

            if(p*max > thresh) {
                p *= max;
                group = this.child[max_i];

                if(this.child[max_i] < 0) {
                    return max_i;
                }
            }
            else if(group == 0) {
                return max_i;
            }
            else {
                return this.parent[this.groupOffset[group]];
            }
        }
    }

    public static Tree readTree(String fileName) {

        Tree t = new Tree();
        int last_parent = -1;
        int groupSize = 0;
        int groups = 0;
        int n = 0;

        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));

            String s;

            List<Integer> _parent = new ArrayList<>();
            List<Integer> _child = new ArrayList<>();
            List<String> _name = new ArrayList<>();
            List<Integer> _groupOffset = new ArrayList<>();
            List<Integer> _groupSize = new ArrayList<>();
            List<Integer> _group = new ArrayList<>();

            while((s = reader.readLine()) != null) {

                String[] ss = s.split(" ");
                int parent = Integer.parseInt(ss[1].strip());
                _parent.add(parent);
                _child.add(-1);
                _name.add(ss[0]);

                if(parent != last_parent) {
                    groups++;
                    _groupOffset.add(n - groupSize);
                    _groupSize.add(groupSize);
                    groupSize = 0;
                    last_parent = parent;
                }
                _group.add(groups);
                if(parent >= 0) {
                    _child.set(parent,groups);
                }
                n++;
                groupSize++;
            }
            reader.close();

            groups++;
            _groupOffset.add(n - groupSize);
            _groupSize.add(groupSize);

            t.name = (String[]) _name.toArray();
            t.group = IntegerToIntArray(_group);
            t.parent = IntegerToIntArray(_parent);
            t.child = IntegerToIntArray(_child);
            t.groupOffset = IntegerToIntArray(_groupOffset);
            t.groupSize = IntegerToIntArray(_groupSize);
            t.n = n;
            t.groups = groups;
            t.leaf = new int[n];

            for(int i = 0; i < n; i++) {
                t.leaf[i] = 1;
            }

            for(int i = 0; i < n; i++) {

                if(t.parent[i] >= 0) {
                    t.leaf[t.parent[i]] = 0;
                }
            }

            return t;
        }
        catch (Exception e) {
            System.out.println(String.format("Error trying to read tree from file '%s'.",fileName));
            return null;
        }
    }

    private static int[] IntegerToIntArray(List<Integer> list) {

        int[] array = new int[list.size()];

        for(int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }

        return array;
    }
}
