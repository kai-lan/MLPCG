#pragma once

#include "zlib.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace IO{

struct ZipFileHeader
{
	unsigned short version;
	unsigned short flags;
	unsigned short compression_type;
	unsigned short stamp_date, stamp_time;
	unsigned int crc;
	unsigned int compressed_size, uncompressed_size;
	std::string filename;
	unsigned int header_offset; // local header offset

	ZipFileHeader()
	{}

	ZipFileHeader(const std::string& filename_input)
		:version(20), flags(0), compression_type(8), stamp_date(0), stamp_time(0), crc(0),
		compressed_size(0), uncompressed_size(0), filename(filename_input), header_offset(0)
	{}

	bool Read(std::istream& istream, const bool global, std::string& err_msg)
	{
		unsigned int sig;
		// read and check for local/global magic
		if (global) {
			ReadPrimitive(istream, sig);
			if (sig != 0x02014b50) 
			{ 
				//std::cerr << "Did not find global header signature" << std::endl; 
				err_msg += "ZipFileHeader: Did not find global header signature.\n";
				return false; 
			}
			ReadPrimitive(istream, version);
		}
		else 
		{
			ReadPrimitive(istream, sig);
			if (sig != 0x04034b50) 
			{ 
				//LOG::cerr << "Did not find local header signature" << std::endl; 
				err_msg += "ZipFileHeader: Did not find local header signature.\n";
				return false; 
			}
		}
		// Read rest of header
		ReadPrimitive(istream, version);
		ReadPrimitive(istream, flags);
		ReadPrimitive(istream, compression_type);
		ReadPrimitive(istream, stamp_date);
		ReadPrimitive(istream, stamp_time);
		ReadPrimitive(istream, crc);
		ReadPrimitive(istream, compressed_size);
		ReadPrimitive(istream, uncompressed_size);
		unsigned short filename_length, extra_length;
		ReadPrimitive(istream, filename_length);
		ReadPrimitive(istream, extra_length);
		unsigned short comment_length = 0;
		if (global) {
			ReadPrimitive(istream, comment_length); // filecomment
			unsigned short disk_number_start, int_file_attrib;
			unsigned int ext_file_attrib;
			ReadPrimitive(istream, disk_number_start); // disk# start
			ReadPrimitive(istream, int_file_attrib); // internal file
			ReadPrimitive(istream, ext_file_attrib); // ext final
			ReadPrimitive(istream, header_offset);
		} // rel offset
		char* buf = new char[std::max(comment_length, std::max(filename_length, extra_length))];
		istream.read(buf, filename_length);
		buf[filename_length] = 0;
		filename = std::string(buf);
		istream.read(buf, extra_length);
		if (global) istream.read(buf, comment_length);
		delete[] buf;
		return true;
	}

protected:
	template<class T>
	void
	ReadPrimitive(std::istream& input, T& d)
	{
		//STATIC_ASSERT((sizeof(T) == PLATFORM_INDEPENDENT_SIZE<T>::value));
		input.read((char*)&d, sizeof(T));
		//if (big_endian) Swap_Endianity(d); // convert to little endian if necessary
	}
};

struct GZipFileHeader
{
	unsigned char magic0, magic1; // magic should be 0x8b,0x1f
	unsigned char cm; // compression method 0x8 is gzip
	unsigned char flags; // flags
	unsigned int modtime; // 4 byte modification time
	unsigned char flags2; // secondary flags
	unsigned char os; // operating system 0xff for unknown
	unsigned short crc16; // crc check
	unsigned int crc32;

	GZipFileHeader()
		:magic0(0), magic1(0), flags(0), modtime(0), flags2(0), os(0), crc16(0), crc32(0)
	{}

	bool Read(std::istream& istream, std::string& err_msg)
	{
		ReadPrimitive(istream, magic0);
		ReadPrimitive(istream, magic1);
		if (magic0 != 0x1f || magic1 != 0x8b) 
		{ 
			//LOG::cerr << "gzip: did not find gzip magic 0x1f 0x8b" << std::endl;
			err_msg += "gzip: did not find gzip magic 0x1f 0x8b";
			return false; 
		}
		ReadPrimitive(istream, cm);
		if (cm != 8) 
		{ 
			//LOG::cerr << "gzip: compression method not 0x8" << std::endl;
			err_msg += "gzip: compression method not 0x8";
			return false; 
		}
		ReadPrimitive(istream, flags);
		ReadPrimitive(istream, modtime);
		ReadPrimitive(istream, flags2);
		ReadPrimitive(istream, os);
		unsigned char dummyByte;
		// read flags if necessary
		if (flags & 2) {
			unsigned short flgExtraLen;
			ReadPrimitive(istream, flgExtraLen);
			for (int k = 0; k < flgExtraLen; k++) 
				ReadPrimitive(istream, dummyByte);
		}
		// read filename/comment if present
		int stringsToRead = ((flags & 8) ? 1 : 0) + ((flags & 4) ? 1 : 0);
		for (int i = 0; i < stringsToRead; i++)
			do { ReadPrimitive(istream, dummyByte); } while (dummyByte != 0 && istream);
		if (flags & 1) ReadPrimitive(istream, crc16);
		if (!istream) 
		{ 
			//LOG::cerr << "gzip: got to end of file after only reading gzip header" << std::endl;
			err_msg += "gzip: got to end of file after only reading gzip header";
			return false; 
		}
		return true;
	}
protected:
	template<class T>
	void
		ReadPrimitive(std::istream& input, T& d)
	{
		//STATIC_ASSERT((sizeof(T) == PLATFORM_INDEPENDENT_SIZE<T>::value));
		input.read((char*)&d, sizeof(T));
		//if (big_endian) Swap_Endianity(d); // convert to little endian if necessary
	}
};

class ZipStreambufDecompress : public std::streambuf
{
	static const unsigned int buffer_size = 512;
	std::istream& istream;

	z_stream strm;
	unsigned char in[buffer_size], out[buffer_size];
	ZipFileHeader header;
	GZipFileHeader gzip_header;
	int total_read, total_uncompressed;
	bool part_of_zip_file;
	bool valid;
	bool compressed_data;

	static const unsigned short DEFLATE = 8;
	static const unsigned short UNCOMPRESSED = 0;
public:
	ZipStreambufDecompress(std::istream& stream, bool part_of_zip_file_input)
		:istream(stream), total_read(0), total_uncompressed(0), part_of_zip_file(part_of_zip_file_input), valid(true)
	{
		strm.zalloc = Z_NULL; strm.zfree = Z_NULL; strm.opaque = Z_NULL; strm.avail_in = 0; strm.next_in = Z_NULL;
		setg((char*)in, (char*)in, (char*)in);
		setp(0, 0);
		// skip the header
		std::string err_msg;
		if (part_of_zip_file) {
			valid = header.Read(istream, false, err_msg);
			if (header.compression_type == DEFLATE) compressed_data = true;
			else if (header.compression_type == UNCOMPRESSED) compressed_data = false;
			else {
				compressed_data = false; 
				//LOG::cerr << "ZIP: got unrecognized compressed data (Supported deflate/uncompressed)" << std::endl;
				std::cerr << "ZIP: got unrecognized compressed data (Supported deflate/uncompressed)" << std::endl;
				valid = false;
			}
		}
		else { 
			valid = gzip_header.Read(istream, err_msg); 
			compressed_data = true; 
		}
		// initialize the inflate
		if (compressed_data && valid) {
			int result = inflateInit2(&strm, -MAX_WBITS);
			if (result != Z_OK) 
			{ 
				//LOG::cerr << "gzip: inflateInit2 did not return Z_OK" << std::endl;
				std::cerr << "gzip: inflateInit2 did not return Z_OK" << std::endl;
				valid = false; 
			}
		}
	}

	virtual ~ZipStreambufDecompress()
	{
		if (compressed_data && valid) inflateEnd(&strm);
		if (!part_of_zip_file) delete& istream;
	}

	int Process()
	{
		if (!valid) return -1;
		if (compressed_data) {
			strm.avail_out = buffer_size - 4;
			strm.next_out = (Bytef*)(out + 4);
			while (strm.avail_out != 0) {
				if (strm.avail_in == 0) { // buffer empty, read some more from file
					istream.read((char*)in, part_of_zip_file ? std::min((unsigned int)buffer_size, header.compressed_size - total_read) : (unsigned int)buffer_size);
					strm.avail_in = (uInt)istream.gcount();
					total_read += strm.avail_in;
					strm.next_in = (Bytef*)in;
				}
				int ret = inflate(&strm, Z_NO_FLUSH); // decompress
				switch (ret) {
				case Z_STREAM_ERROR:
					//LOG::cerr << "libz error Z_STREAM_ERROR" << std::endl;
					std::cerr << "libz error Z_STREAM_ERROR" << std::endl;
					valid = false; return -1;
				case Z_NEED_DICT:
				case Z_DATA_ERROR:
				case Z_MEM_ERROR:
					//LOG::cerr << "gzip error " << strm.msg << std::endl;
					std::cerr << "gzip error " << strm.msg << std::endl;
					valid = false; return -1;
				}
				if (ret == Z_STREAM_END) break;
			}
			int unzip_count = buffer_size - strm.avail_out - 4;
			total_uncompressed += unzip_count;
			return unzip_count;
		}
		else { // uncompressed, so just read
			istream.read((char*)(out + 4), std::min(buffer_size - 4, header.uncompressed_size - total_read));
			int count = (int)istream.gcount();
			total_read += count;
			return count;
		}
		return 1;
	}

	virtual int underflow() // this is an std function override; don't rename
	{
		if (gptr() && (gptr() < egptr())) return traits_type::to_int_type(*gptr()); // if we already have data just use it
		int put_back_count = (int)(gptr() - eback());
		if (put_back_count > 4) put_back_count = 4;
		std::memmove(out + (4 - put_back_count), gptr() - put_back_count, put_back_count);
		int num = Process();
		setg((char*)(out + 4 - put_back_count), (char*)(out + 4), (char*)(out + 4 + num));
		if (num <= 0) return EOF;
		return traits_type::to_int_type(*gptr());
	}

	virtual int overflow(int c = EOF) // this is an std function override; don't rename
	{
		assert(false); 
		return EOF;
	}
};

// Class needed because istream cannot own its streambuf
class ZipFileIstream : public std::istream
{
	ZipStreambufDecompress buf;
public:
	ZipFileIstream(std::istream& istream, bool part_of_zip_file)
		: std::istream(&buf)
		, buf(istream, part_of_zip_file)
	{}

	virtual ~ZipFileIstream()
	{}
};

class ZipFileReader
{
	std::ifstream istream;
public:
	//HASHTABLE<std::string, ZipFileHeader*> filename_to_header;
	std::unordered_map<std::string, std::shared_ptr<ZipFileHeader>> filename_to_header;

	ZipFileReader(const std::string& filename)
	{
		istream.open(filename.c_str(), std::ios::in | std::ios::binary);
		if (!istream)
		{
#if PLATFORM_EXCEPTIONS_DISABLED
			return;
#else
			//throw FILESYSTEM_ERROR("ZIP: Invalid file handle");
			//throw std::exception("ZIP: Invalid file handle");
			throw std::filesystem::filesystem_error(std::string("ZIP: Invalid file handle"), filename, std::error_code());
#endif
		}

		std::string err_msg;
		FindAndReadCentralHeader(err_msg);
	}

	virtual ~ZipFileReader()
	{}

	std::istream* GetFile(const std::string& filename, const bool binary = true)
	{
		//ZipFileHeader** header = filename_to_header.Get_Pointer(filename);
		auto it=filename_to_header.find(filename);
		if(it != filename_to_header.end())
		{
			//ZipFileHeader** header = &filename_to_header[filename].get();
			if(ZipFileHeader* header = it->second.get())
			{ 
				istream.seekg(header->header_offset); 
				return new ZipFileIstream(istream, true); 
			}
		}
		return nullptr;
	}

	void GetFileList(std::vector<std::string>& filenames) const
	{
		filenames.reserve(filename_to_header.size());
		for(const auto& kv : filename_to_header)
			filenames.push_back(kv.first);
	}
private:
	template<class T> 
	void
	ReadPrimitive(std::istream& input, T& d)
	{
		//STATIC_ASSERT((sizeof(T) == PLATFORM_INDEPENDENT_SIZE<T>::value));
		input.read((char*)&d, sizeof(T));
		//if (big_endian) Swap_Endianity(d); // convert to little endian if necessary
	}

	bool 
	FindAndReadCentralHeader(std::string& err_msg)
	{
		// Find the header
		// NOTE: this assumes the zip file header is the last thing written to file...
		istream.seekg(0, std::ios_base::end);
		//int end_position = istream.tellg();
		std::streamoff end_position = istream.tellg();
		unsigned int max_comment_size = 0xffff; // max size of header
		unsigned int read_size_before_comment = 22;
		std::streamoff read_start = max_comment_size + read_size_before_comment;
		if (read_start > end_position) read_start = end_position;
		istream.seekg(end_position - read_start);
		if (read_start <= 0) 
		{ 
			//LOG::cerr << "ZIP: Invalid read buffer size" << std::endl;
			err_msg += "ZipFileReader: Invalid read buffer size.\n";
			return false; 
		}
		char* buf = new char[(unsigned int)read_start];
		istream.read(buf, read_start);
		int found = -1;
		for (unsigned int i = 0; i < read_start - 3; i++) {
			if (buf[i] == 0x50 && buf[i + 1] == 0x4b && buf[i + 2] == 0x05 && buf[i + 3] == 0x06) 
			{ 
				found = i; 
				break; 
			}
		}
		delete[] buf;
		if (found == -1) 
		{ 
			//LOG::cerr << "ZIP: Failed to find zip header" << std::endl;
			err_msg += "ZipFileReader: Failed to find zip header.\n";
			return false; 
		}
		// seek to end of central header and read
		istream.seekg(end_position - (read_start - found));
		unsigned int word;
		unsigned short disk_number1, disk_number2, num_files, num_files_this_disk;
		ReadPrimitive(istream, word); // end of central
		ReadPrimitive(istream, disk_number1); // this disk number
		ReadPrimitive(istream, disk_number2); // this disk number
		if (disk_number1 != disk_number2 || disk_number1 != 0) {
			//LOG::cerr << "ZIP: multiple disk zip files are not supported" << std::endl;
			err_msg += "ZipFileReader: multiple disk zip files are not supported.\n";
			return false;
		}
		ReadPrimitive(istream, num_files); // one entry in center in this disk
		ReadPrimitive(istream, num_files_this_disk); // one entry in center 
		if (num_files != num_files_this_disk) {
			//LOG::cerr << "ZIP: multi disk zip files are not supported" << std::endl;
			err_msg += "ZIP: multi disk zip files are not supported.\n";
			return false;
		}
		unsigned int size_of_header, header_offset;
		ReadPrimitive(istream, size_of_header); // size of header
		ReadPrimitive(istream, header_offset); // offset to header
		// go to header and read all file headers
		istream.seekg(header_offset);
		for (int i = 0; i < num_files; i++) 
		{
			std::shared_ptr<ZipFileHeader> header(new ZipFileHeader);
			if (header->Read(istream, true, err_msg)) 
				filename_to_header.insert(std::make_pair(header->filename, header));
		}
		return true;
	}

};

} //namespace IO

