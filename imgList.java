package com.example.android.Application;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;

import javax.imageio.ImageIO;

public class imgList implements Serializable 
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	ArrayList<byte[]> imageList;
	ArrayList<String> textList;
	
	transient BufferedImage img;
	File file;
	FileWriter fileWrite;
	public imgList(ArrayList<byte[]> i, ArrayList<String> t)
	{
		imageList = i;
		textList = t;
	}
	void showimg()
	{
		try
		{
		
			for(int i = 0; i<imageList.size(); i++)
			{
				System.out.println("img"+i+" "+imageList);
				img = ImageIO.read(new ByteArrayInputStream(imageList.get(i)));
				ImageIO.write(img,"jpg",new File("./img_out/img"+i+".jpg"));
				file = new File("./img_out/img"+i+".txt");
				fileWrite = new FileWriter(file, true);
				fileWrite.write(textList.get(i));
				fileWrite.flush();		
	
			}
			fileWrite.close();
		}
		catch (IOException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
