public class SLlist<T>{
    private Node head;
    private Node tail;
    private int size;

    private class Node{
        T data;
        Node next;

        Node(T data){
            this(data, null);
        }
        Node(T data, Node next){
            this.data = data;
            this.next = next;
        }
    }

    SLlist(){
        this.head = null;
        this.tail = null;
        this.size = 0;
    }

    private Node getNode(int index){
        if(index < 0 || index >= this.size) throw new IndexOutOfBoundsException();
        if(index == 0)  return this.head;

        Node x = head;
        while(--index >= 0){
            x = x.next;
        }
        return x;
    }
    //뒤에 추가
    public boolean add(T data){
        this.tail.next = new Node(data);
        this.tail = this.tail.next;
        this.size++;
        if(this.size == 0)  this.tail = this.head;
        return true;
    }

    public boolean add(T data, int index){
        if(index < 0 || index > this.size)  throw new IndexOutOfBoundsException();
        if(index == this.size)  return add(data);
        if(index == 0){
            head = new Node(data, head.next);
            if(this.size == 0)  this.tail = this.head;
            this.size++;
            return true;
        }
        Node prevNode = getNode(index);
        prevNode.next = new Node(data, prevNode.next);
        this.size++;
        return true;
    }
    
    //뒤에삭제?

    public void remove(){
    }
//    public T remove(int index){
//        if(index < 0 || index >= this.size) throw new IndexOutOfBoundsException();
//        if(index == this.size-1){
//            Node rmNode = this.tail;
//            this.tail.data = null;
//            this.tail.next = null;
//            this.tail = getNode(index-1);
//            return rmNode.data;
//
//        }
//        return T;
//    }
    
    //index위치 반환
    public T get(int index){
        if(index < 0 || index >= this.size) throw new IndexOutOfBoundsException();
        if(index == 0)  return this.head.data;
        if(index == this.size-1)    return this.tail.data;
        return getNode(index).data;
    }


    public int size(){
        return this.size;
    }
    public void clear(){
        for(Node x = head; x != null;){
            Node nextNode = x.next;
            x.data = null;
            x.next = null;
            x = nextNode;
        }
        this.head = this.tail = null;
        this.size = 0;
    }

}

/*
public class SLlist<T>
public Node getNode(int index)
public void add()
public void addFirst(T value)
public void addLast(T value)
public T remove()
public T get()
public int size()
public void clear()
 */
