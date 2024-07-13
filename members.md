---
layout: page
---
<script setup>
import {
  VPTeamPage,
  VPTeamPageTitle,
  VPTeamMembers
} from 'vitepress/theme'

const members = [
  {
    avatar: 'https://avatars.githubusercontent.com/u/112888202',
    name: 'kiyotakali',
    links: [
      { icon: 'github', link: 'https://github.com/kiyotakali' },
    ]
  },
  {
    avatar: "https://avatars.githubusercontent.com/u/55980292",
    name: "yy4382",
    links: [
      { icon: 'github', link: 'https://github.com/yy4382' },
    ]
  },
  {
    avatar: "https://avatars.githubusercontent.com/u/123150988?v=4",
    name: "xjtu-wjz",
    links: [
      { icon: 'github', link: 'https://github.com/xjtu-wjz' },
    ]
  },
  {
    avatar: "https://avatars.githubusercontent.com/u/107924172?v=4",
    name: "xjtu-cch",
    links: [
      { icon: 'github', link: 'https://github.com/2421468125' },
    ]
  },
  {
    avatar: "https://avatars.githubusercontent.com/u/140293041?v=4",
    name: "Lenny-Dai",
    links: [
      { icon: 'github', link: 'https://github.com/Lenny-Dai' },
    ]
  },
  {
    avatar: "https://avatars.githubusercontent.com/u/121414835?v=4",
    name: "worfsmile",
    links: [
      { icon: 'github', link: 'https://github.com/worfsmile' },
    ]
  }
]
</script>

<VPTeamPage>
  <VPTeamPageTitle>
    <template #title>
      Members
    </template>
    <template #lead>
      Members of our group.
    </template>
  </VPTeamPageTitle>
  <VPTeamMembers
    :members="members"
  />
</VPTeamPage>