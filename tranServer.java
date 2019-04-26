package com.example.android.Application;



import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.util.ArrayList;
import java.io.*;

public class tranServer extends Thread {
	public static ServerSocket sock;
	public tranServer(ServerSocket s)
	{
		sock = s;
	}
	
    @Override
	public void run() {
		// TODO Auto-generated method stub
//       while(true){
        try {

            Socket socket = sock.accept();
			System.out.println("socket accepted");
            ObjectInputStream ois = new ObjectInputStream(socket.getInputStream());
            transform trans = (transform)ois.readObject();
            trans.BtoI();
            Thread.sleep(5000);
    		Send send = new Send(socket,sock);
    		send.start();

			
//			oos.close();
//			Thread.sleep(4000);
//          ois.close();
		
//            socket.close();
//            sock.close();

            
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
			
        } catch (InterruptedException e) {
			e.printStackTrace();
		}
		
	   }
//	}

	public static void main(String[] args) throws IOException {
	       	// TODO Auto-generated method stub
		
        int port = 8080;
        
        sock = new ServerSocket(port);
		tranServer start = new tranServer(sock);
        start.start();
        
        
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
	public void run() {
	// TODO Auto-generated method stub
		System.out.println("sending started");
        File[] files = new File("./img_out").listFiles();
        byte[] fileContent;
        ArrayList<byte[]> imgByte = new ArrayList<>();
        ArrayList<String> text = new ArrayList<>();
        int fileLen = files.length;
        try 
        {
			for(int i=0; i<fileLen ; i++)
			{
				
				fileContent = Files.readAllBytes(files[i].toPath());				
				imgByte.add(fileContent);
				text.add(files[i].getAbsolutePath());
			}
			
			imgList an = new imgList(imgByte, text);
			Object obj = (Object)an;
			ObjectOutputStream oos = new ObjectOutputStream(s.getOutputStream());
			oos.writeObject(obj);

			Receive r = new Receive(serverSock);
			r.start();
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
	public Receive(ServerSocket s)
	{
		serverSock = s;
	}
	
	Socket socket;

	InputStream is;
	InputStreamReader isr;
	BufferedReader br;
	File file;
	FileWriter writer;
	public void run()
	{
		try{
		
		socket = serverSock.accept();
		is = socket.getInputStream();
		isr = new InputStreamReader(is);
		br = new BufferedReader(isr);
		String data = br.readLine();
		File file = new File("./img/rating.txt");
		writer = new FileWriter(file, true);
		writer.write(data);
		writer.flush();
		writer.close();

		serverSock.close();
		}catch(IOException e)
		{
			e.printStackTrace();
		}
	}
}
