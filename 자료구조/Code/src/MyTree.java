import java.util.LinkedList;
import java.util.Queue;

public class MyTree<T> {
    private class Node{
        T data;
        Node left, right;
        Node(T data){
            this.data = data;
            left = null;
            right = null;
        }
    }

    private Node root;
    private int node_num;

    MyTree(){
        this.root = null;
        this.node_num = 0;
    }

    void insert(T data){
        Node newNode = new Node(data);
        if(this.root == null)   {
            root = newNode;
        }

        else if((node_num & (node_num + 1)) == 0){
            System.out.println("full");
            Node curNode =root;
            while(curNode.left != null) curNode = curNode.left;
            curNode.left = newNode;
        }
        else{
            Queue<Node> q = new LinkedList<>();
            q.add(root);
            while(!q.isEmpty()){
                Node curNode = q.poll();
                if(curNode.left == null){
                    curNode.left = newNode;
                    break;
                }
                else if(curNode.right == null){
                    curNode.right = newNode;
                    break;
                }
                else{
                    q.add(curNode.left);
                    q.add(curNode.right);
                }
            }
        }
        node_num+=1;
    }

    private void preorder(Node startNode) {
        if (startNode != null) {
            System.out.println(startNode.data.toString());
            preorder(startNode.left);
            preorder(startNode.right);
        }
    }
    private void inorder(Node startNode) {
        if (startNode != null) {
            inorder(startNode.left);
            System.out.println(startNode.data.toString());
            inorder(startNode.right);
        }
    }
    private void postorder(Node startNode) {
        if (startNode != null) {
            postorder(startNode.left);
            postorder(startNode.right);
            System.out.println(startNode.data.toString());
        }
    }

    void preorder() {
        preorder(root);
    }
    void inorder() {
        inorder(root);
    }
    void postorder() {
        postorder(root);
    }

}
