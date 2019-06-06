package com.example.android.Application;



import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.*;
import java.util.ArrayList;
import java.io.*;

public class tranServer extends Thread {
	public static ServerSocket sock;
	public Socket android;
	public static Socket python_client;
	public static int count = 0;
	String dir_string;
	private transform trans;
	public tranServer(transform trans ,Socket android)
	{
		this.trans = trans;
		this.android = android;
	}
	
    @Override
	public void run() {
		// TODO Auto-generated method stub

        try {

	            dir_string = ""+android.getInetAddress()+":"+android.getPort();
	            trans.setDir(dir_string);
	            trans.BtoI();
	            
	            python_client = new Socket("localhost",8008);
	            PrintWriter out = new PrintWriter(python_client.getOutputStream(), true);
	            out.print("/home/root/coordi/testing/"+dir_string);
	            out.flush();
	            
	            BufferedReader stdIn =new BufferedReader(new InputStreamReader(python_client.getInputStream()));
	            String in = stdIn.readLine();
	            System.out.println("image_set set. "+in);
	            
	    		Send send = new Send(android,sock);
	    		send.start();

			
//			oos.close();
//			Thread.sleep(4000);
//    		ois.close();
		
//          socket.close();
//          sock.close();

            
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
		
	   
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException {
	       	// TODO Auto-generated method stub
		
        int port = 8080;
        
        sock = new ServerSocket(port);
        while(true)
        {
        	System.out.println("waiting client");
	        Socket client = sock.accept();
			count++;
			System.out.println("socket accepted ->,"+client+" number of counts = "+count);
			System.out.println("address -> "+client.getInetAddress()+"\n"+"port-> "+client.getPort());
            ObjectInputStream ois = new ObjectInputStream(client.getInputStream());
            transform trans = (transform)ois.readObject();
            if (trans.identifier == 0)
            {
    			tranServer start = new tranServer(trans, client);
    	        start.start();
            }
            else if (trans.identifier == 1)
            {
            	String rating = trans.rate;
    			Receive r = new Receive(rating);
    			r.start();
            }

        }
        
    }

}

class Send extends Thread
{
	ServerSocket serverSock;
	Socket s;
	public Send(Socket socket, ServerSocket sock)
	{
		s = socket;
		serverSock = sock;
	}
	@Override
	public void run() 
	{
	// TODO Auto-generated method stub
		System.out.println("sending started");
        File[] files = new File("/home/root/coordi/recommend").listFiles();
        byte[] fileContent;
        ArrayList<byte[]> imgByte = new ArrayList<>();
        ArrayList<String> text = new ArrayList<>();
        int fileLen = files.length;
        Path path = Paths.get("/home/root/coordi/after_recommend");
        try 
        {
			for(int i=0; i<fileLen ; i++)
			{
				
				fileContent = Files.readAllBytes(files[i].toPath());				
				imgByte.add(fileContent);
				text.add(files[i].getName());
				Files.move(files[i].toPath(), path.resolve(files[i].getName()), StandardCopyOption.REPLACE_EXISTING);
			}
			
			imgList an = new imgList(imgByte, text);
			Object obj = (Object)an;
			ObjectOutputStream oos = new ObjectOutputStream(s.getOutputStream());
			oos.writeObject(obj);
			s.close();

//			oos.flush();
//			Thread.sleep(4000);
//			oos.close();
//			s.close();
			//s.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}/* catch (InterruptedException e) {
			e.printStackTrace();
		}*/
	}
	
	
	
}

class Receive extends Thread
{
	ServerSocket serverSock;
	String rate;
	public Receive(String rate)
	{
		this.rate = rate;
	}
	File file;
	FileWriter writer;
	public void run()
	{
		try
		{
			File file = new File("/home/root/coordi/rating/rating.txt");
			writer = new FileWriter(file);
			writer.write(rate);
			writer.flush();
			writer.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}
}
