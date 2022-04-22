public class SLlist<T>{
    Node head, tail;
    int size;

    class Node{
        T data;
        Node next;
        Node(T data){
            this.data = data;
            this.next = null;
        }
    }

    SLlist(){
        this.head = null;
        this.tail = null;
        this.size = 0;
    }

    private Node getNode(int index){
        if(index < 0 || index >= this.size) throw new IndexOutOfBoundsException();  //out of bounds
        if(index == 0)  return this.head;
        if(index == this.size-1)  return this.tail;

        Node node = head;
        for(int i= 0; i < index; i++){
            node = node.next;
        }

        return node;
    }
    public boolean add(T data){
        if(this.head == null)   addFirst(data); //리스트가 비어있을때 앞에 넣는다
        else addLast(data);
        this.size++;
        return true;
    }
    public void addFirst(T data){
        Node newNode = new Node(data);
        head = newNode;

        if(this.size == 0)  this.tail = head;
    }

    public void addLast(T data){
        Node newNode = new Node(data);
        tail.next = newNode;
        tail = newNode;
    }

    public T remove(){

    }
    public T remove(int index){
        if(index < 0 || index >= this.size) throw new IndexOutOfBoundsException();
        if(index == 0)  removeFirst();
        else if(index == this.size-1)   removeLast();
        else{
            Node prevNode = getNode(index);
            Node removeNode = prevNode.next;
            prevNode.next = prevNode.next.next;
            return removeNode.data;
        }
    }

    public T get(int index){
        if(index < 0 || index >= this.size) throw new IndexOutOfBoundsException();
        return getNode(index).data;
    }
    private Node removeFirst(){
        Node removeNode = head;
        head.data = null;
        head.next = null;
        return removeNode;
    }
    private T removeLast(){

    }



    public int size(){
        return this.size;
    }
    public void clear(){
        Node x = head;
        while(x != null){
            Node nextNode = x.next;
            x.data = null;
            x.next = null;
            x = nextNode;
        }
    }
}
/*
public Node getNode(int index)
public void add()
public void addFirst(T value)
public void addLast(T value)
public T remove()
public T get()
public int size()
public void clear()
 */
