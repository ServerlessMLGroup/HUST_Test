  int offset=0;
  while(flag[0]==0)
  {
    __nanosleep(100);
  }

  if((blocknum[0]*blocknum[1]*blocknum[2])>blocksize[0])
  {
    offset=vx;
    while(offset<(blocknum[0]*blocknum[1]*blocknum[2]))
    {
    vz=(offset-1)/(blocknum[0]*blocknum[1]);
    vy= (offset-(vz*blocknum[0]*blocknum[1])-1)/blocknum[0];
    vx=offset - (vz*blocknum[0]*blocknum[1])-vy*blocknum[0]-1;

    offset+=blocksize[0];
    }
  }
  else
  {

  }